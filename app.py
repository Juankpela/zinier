import asyncio
import base64
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import time
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import anthropic
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-20250514"
).strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

API_PORT = int(os.getenv("PORT", "8001"))
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "1568"))
MIN_IMAGE_EDGE = int(os.getenv("MIN_IMAGE_EDGE", "120"))
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "8"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
PATTERN_CACHE_SECONDS = int(os.getenv("PATTERN_CACHE_SECONDS", "300"))

# Verificacion opcional en segunda pasada (solo se dispara si el primer
# analisis queda marcado para revision). Apagada por defecto para no
# duplicar costo/latencia. VERIFICATION_MODEL permite escalar a un modelo
# mas potente (o a otro proveedor en el futuro) sin tocar codigo.
ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "false").strip().lower() in {"1", "true", "yes", "on"}
VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", ANTHROPIC_MODEL).strip()
# Tolerancia de cables faltantes antes de declarar inconsistencia
# (un cable puede ocluir parcialmente a otro en angulos cerrados).
OCCUPANCY_CABLE_TOLERANCE = int(os.getenv("OCCUPANCY_CABLE_TOLERANCE", "1"))

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_fibra_optica")

supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    from supabase import create_client

    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase conectado.")
else:
    logger.warning("Supabase no configurado. El aprendizaje estara desactivado.")

anthropic_client = (
    anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    if ANTHROPIC_API_KEY
    else None
)
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
rate_limit_lock = asyncio.Lock()
request_log: dict[str, list[float]] = {}
patterns_cache: dict[str, Any] = {"expires_at": 0.0, "patterns": []}

app = FastAPI(
    title="API Fibra Optica MVP",
    description="Analiza imagenes de nodos ATP con Claude Vision y validacion estructurada.",
    version="5.1.0-mvp",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


def parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


cors_origins = parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


class ImageInput(BaseModel):
    image: str


class FeedbackInput(BaseModel):
    pattern: str


def require_token(request: Request) -> None:
    if not API_ACCESS_TOKEN:
        return
    auth_header = request.headers.get("authorization", "")
    api_key = request.headers.get("x-api-key", "")
    bearer = auth_header.removeprefix("Bearer ").strip()
    if bearer != API_ACCESS_TOKEN and api_key != API_ACCESS_TOKEN:
        raise HTTPException(401, "Token de API invalido.")


async def enforce_rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60
    async with rate_limit_lock:
        timestamps = [t for t in request_log.get(client_ip, []) if t >= window_start]
        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(429, "Demasiadas solicitudes. Intenta de nuevo en un minuto.")
        timestamps.append(now)
        request_log[client_ip] = timestamps


def is_public_ip(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise HTTPException(400, "No se pudo resolver el host de la imagen.")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def validate_image_url(image_url: str) -> str:
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(400, "La URL de imagen debe usar http o https.")
    if not parsed.hostname:
        raise HTTPException(400, "La URL de imagen no tiene host valido.")
    if not is_public_ip(parsed.hostname):
        raise HTTPException(400, "La URL de imagen apunta a una red no permitida.")
    return image_url


async def download_image(image_url: str) -> tuple[bytes, str]:
    validate_image_url(image_url)
    timeout = httpx.Timeout(20.0, connect=5.0)
    headers = {"User-Agent": "api-zinier-fibra/5.0"}

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            async with client.stream("GET", image_url, headers=headers) as response:
                if 300 <= response.status_code < 400:
                    raise HTTPException(400, "La URL redirige; envia la URL final de la imagen.")
                response.raise_for_status()

                content_type = response.headers.get("content-type", "image/jpeg").split(";")[0].lower()
                if content_type not in ALLOWED_IMAGE_TYPES:
                    raise HTTPException(400, "El recurso descargado no parece ser una imagen soportada.")

                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "La imagen supera el tamano maximo permitido.")

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise HTTPException(413, "La imagen supera el tamano maximo permitido.")
                    chunks.append(chunk)
                return b"".join(chunks), content_type
    except httpx.TimeoutException:
        raise HTTPException(504, "Timeout descargando la imagen.")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(400, f"No se pudo descargar la imagen: HTTP {exc.response.status_code}")


def preprocess_image(image_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "La imagen supera el tamano maximo permitido.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "El archivo no es una imagen valida.")

    width, height = image.size
    if min(width, height) < MIN_IMAGE_EDGE:
        raise HTTPException(400, "La imagen es demasiado pequena para analizar puertos con confianza.")

    original_format = image.format or "unknown"
    image = image.convert("RGB")
    longest_edge = max(width, height)
    resized = False
    if longest_edge > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / longest_edge
        new_size = (round(width * scale), round(height * scale))
        image = image.resize(new_size, Image.LANCZOS)
        resized = True

    output = BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)
    processed = output.getvalue()
    if len(processed) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "La imagen procesada supera el tamano maximo permitido.")

    metadata = {
        "original_width": width,
        "original_height": height,
        "processed_width": image.size[0],
        "processed_height": image.size[1],
        "original_format": original_format,
        "processed_media_type": "image/jpeg",
        "resized": resized,
        "sha256": hashlib.sha256(processed).hexdigest(),
    }
    return processed, metadata


def check_known_image(image_key: str):
    if not supabase_client:
        return None
    try:
        result = (
            supabase_client.table("port_corrections")
            .select("correct_result")
            .eq("image_key", image_key)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            return result.data[0]["correct_result"]
    except Exception as exc:
        logger.warning("Error verificando imagen conocida: %s", exc)
    return None


def get_visual_patterns():
    now = time.monotonic()
    if patterns_cache["expires_at"] > now:
        return patterns_cache["patterns"]
    if not supabase_client:
        return []
    try:
        result = (
            supabase_client.table("visual_patterns")
            .select("pattern")
            .eq("active", True)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        patterns = [r["pattern"] for r in (result.data or []) if r.get("pattern")]
        patterns_cache["patterns"] = patterns
        patterns_cache["expires_at"] = now + PATTERN_CACHE_SECONDS
        return patterns
    except Exception as exc:
        logger.warning("Error obteniendo patrones visuales: %s", exc)
        return []


BASE_PROMPT = """Eres un auditor experto en redes FTTH, CTO, NAP y nodos ATP.

====================================================
REGLA DE ORO
============

La presencia de un adaptador SC/APC verde NO constituye evidencia de ocupación.

Un puerto NO debe clasificarse como OCCUPIED únicamente porque:

* Sea verde.
* Tenga adaptador SC/APC.
* Tenga acoplador.
* Tenga tapa.
* Tenga cuerpo de conector visible.

La ocupación debe estar respaldada por evidencia visual clara de una conexión física.

====================================================
OBJETIVO DEL ANÁLISIS
=====================

Analizar imágenes de cajas ATP, CTO, NAP y distribución FTTH.

Determinar con la mayor precisión posible:

* Número total de puertos.
* Estado individual de cada puerto.
* Evidencia visual que respalda la clasificación.
* Calidad de imagen.
* Nivel de confianza.

Minimizar falsos positivos es más importante que maximizar ocupaciones detectadas.

====================================================
DEFINICIÓN DE ESTADOS
=====================

OCCUPIED

Clasificar como OCCUPIED únicamente cuando exista evidencia visual clara de conexión activa.

Evidencias válidas:

* Cable visible conectado al puerto.
* Fibra visible entrando o saliendo.
* Patch cord visible.
* Latiguillo visible.
* Conector insertado con cable claramente conectado.
* Trayectoria visible de la fibra asociada al puerto.

AVAILABLE

Clasificar como AVAILABLE cuando:

* Solo se observa el adaptador SC/APC.
* Solo se observa una tapa protectora.
* El puerto está vacío.
* No existe evidencia visual de cable conectado.
* No existe evidencia visual de fibra conectada.

UNKNOWN

Clasificar como UNKNOWN cuando:

* La imagen está desenfocada.
* Existe oclusión.
* Existe reflejo.
* Existe sombra.
* El puerto está parcialmente visible.
* La resolución no permite confirmar el estado.

====================================================
PROCESO OBLIGATORIO DE VALIDACIÓN
=================================

Para cada puerto visible responder internamente:

1. ¿Veo un cable conectado?
2. ¿Veo una fibra conectada?
3. ¿Veo un patch cord?
4. ¿Veo un latiguillo?
5. ¿Puedo identificar físicamente una conexión?

Si TODAS las respuestas son NO:

status = "available"

Si existe duda:

status = "unknown"

====================================================
VALIDACIÓN CRUZADA
==================

Antes de responder:

Contar:

* Adaptadores visibles.
* Cables visibles.
* Fibras visibles.

Los puertos OCCUPIED deben estar respaldados por evidencia física observable.

Si el número de puertos OCCUPIED es significativamente mayor al número de cables visibles:

Revisar nuevamente la imagen.

====================================================
CONTROL DE FALSOS POSITIVOS
===========================

Es preferible clasificar un puerto como UNKNOWN antes que clasificarlo erróneamente como OCCUPIED.

Nunca asumir ocupación por color.

Nunca asumir ocupación por forma.

Nunca asumir ocupación por presencia de adaptador.

====================================================
CASOS DE REFERENCIA
===================

CASO A

8 adaptadores verdes visibles.
0 cables visibles.

Resultado:

occupied_ports = []
available_ports = [1,2,3,4,5,6,7,8]

CASO B

8 adaptadores verdes visibles.
1 cable visible conectado al puerto 5.

Resultado:

occupied_ports = [5]
available_ports = [1,2,3,4,6,7,8]

CASO C

8 adaptadores visibles.
4 cables visibles asociados a puertos específicos.

Resultado:

occupied_ports = únicamente los puertos con cable visible.

CASO D

Puertos parcialmente ocultos.

Resultado:

status = "unknown"

====================================================
NUMERACIÓN
==========

* Si existe una sola fila, numerar de izquierda a derecha comenzando en 1.
* Si existen dos filas, numerar primero la fila superior y luego la inferior.
* Utilizar números impresos en la caja como referencia principal.
* No inventar puertos.
* Cada puerto debe aparecer exactamente una vez.

====================================================
CONSISTENCIA OBLIGATORIA
========================

Debe cumplirse:

occupied + available + unknown = total_ports

Ningún puerto puede aparecer en más de una categoría.

====================================================
ANÁLISIS DE CALIDAD
===================

Calcular score de calidad de imagen entre 0 y 100.

Flags permitidos:

* blur
* glare
* occlusion
* angle
* low_resolution
* cropped
* dark

====================================================
NEEDS REVIEW
============

needs_review = true cuando:

* Existe algún puerto UNKNOWN.
* Confidence promedio menor a 75.
* Hay oclusión significativa.
* Hay desenfoque significativo.
* Hay contradicción entre cables visibles y ocupaciones detectadas.

====================================================
MÉTRICAS DE CONTROL
===================

Además del análisis principal devolver el bloque "metrics" con:

* visible_connectors: adaptadores/conectores observables.
* visible_cables: cables observables.
* visible_fibers: fibras observables.

Estas métricas deben representar únicamente elementos observables en la imagen.

====================================================
VALIDACIÓN FINAL OBLIGATORIA
============================

Antes de responder:

Verificar nuevamente que cada puerto OCCUPIED tenga una evidencia específica.

Si no puedes explicar visualmente por qué un puerto está ocupado:

Cambiarlo a AVAILABLE o UNKNOWN.

No inventar evidencia.

No inferir conexiones ocultas.

Basarse únicamente en elementos visibles.

====================================================
PATRONES APRENDIDOS
===================

{patterns}

====================================================
FORMATO DE RESPUESTA
====================

Responde únicamente JSON válido, sin texto adicional.

{
"total_ports": 0,
"ports": [
{
"number": 1,
"status": "occupied",
"confidence": 0,
"evidence": "cable visible conectado al puerto"
}
],
"metrics": {
"visible_connectors": 0,
"visible_cables": 0,
"visible_fibers": 0
},
"image_quality": {
"score": 0,
"flags": []
},
"needs_review": false,
"summary": "resumen operativo breve"
}
"""


def build_prompt() -> str:
    patterns = get_visual_patterns()
    if not patterns:
        return BASE_PROMPT.replace("{patterns}", "")

    lines = "Reglas aprendidas en campo, aplicalas solo si son consistentes con la imagen:\n"
    for pattern in patterns:
        lines += f"- {pattern}\n"
    return BASE_PROMPT.replace("{patterns}", lines)


def extract_text_from_message(message: Any) -> str:
    chunks = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


VERIFY_PROMPT = """Eres un auditor de SEGUNDA REVISIÓN de puertos ópticos FTTH/ATP.

Recibes la MISMA imagen y un análisis previo. Tu tarea es auditarlo, no repetirlo.

REGLA DE ORO:
La presencia de un adaptador SC/APC verde NO es evidencia de ocupación.
Un puerto solo es OCCUPIED si hay evidencia visual clara de cable, fibra, patch cord o latiguillo conectado.

Para cada puerto marcado como OCCUPIED en el análisis previo:
- Verifica que exista evidencia física visible específica.
- Si NO puedes explicar visualmente por qué está ocupado, cámbialo a "available" o "unknown".

Verifica también que: occupied + available + unknown = total_ports.
Recalcula el bloque "metrics" (visible_connectors, visible_cables, visible_fibers) según lo observable.

Responde ÚNICAMENTE el JSON corregido, con exactamente el mismo formato del análisis previo, sin texto adicional.

ANÁLISIS PREVIO A AUDITAR:
{previous}
"""


async def _invoke_vision(
    prompt: str,
    image_bytes: bytes,
    media_type: str,
    model: str,
    max_tokens: int = 900,
) -> str:
    if not anthropic_client:
        raise HTTPException(503, "ANTHROPIC_API_KEY no configurado.")

    image_data = base64.b64encode(image_bytes).decode("utf-8")

    async with analysis_semaphore:
        try:
            logger.info(
                "Claude request model=%s media_type=%s image_size=%s",
                model,
                media_type,
                len(image_bytes),
            )

            message = await anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            },
                        ],
                    }
                ],
            )

            logger.info("Claude response OK")

            return extract_text_from_message(message)

        except Exception as exc:
            logger.exception("ERROR COMPLETO CLAUDE")

            raise HTTPException(
                status_code=502,
                detail=f"{type(exc).__name__}: {str(exc)}"
            )


async def call_claude(image_bytes: bytes, media_type: str) -> str:
    return await _invoke_vision(build_prompt(), image_bytes, media_type, ANTHROPIC_MODEL)


async def verify_with_second_pass(
    image_bytes: bytes,
    media_type: str,
    previous: dict[str, Any],
) -> dict[str, Any]:
    snapshot = {
        "total_ports": previous.get("total_ports"),
        "ports": previous.get("ports"),
        "metrics": previous.get("metrics"),
    }
    prompt = VERIFY_PROMPT.replace(
        "{previous}", json.dumps(snapshot, ensure_ascii=False)
    )
    raw_text = await _invoke_vision(prompt, image_bytes, media_type, VERIFICATION_MODEL)
    return parse_json_response(raw_text)


def parse_json_response(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            raise HTTPException(502, "Claude no devolvio JSON valido.")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            raise HTTPException(502, "Claude devolvio JSON invalido.")

    if not isinstance(parsed, dict):
        raise HTTPException(502, "Claude devolvio una estructura inesperada.")
    return parsed


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_result(data: dict[str, Any], image_meta: dict[str, Any], source: str) -> dict[str, Any]:
    ports = []
    seen_numbers: set[int] = set()

    if isinstance(data.get("ports"), list):
        for item in data["ports"]:
            if not isinstance(item, dict):
                continue
            number = to_int(item.get("number"))
            if number <= 0 or number in seen_numbers:
                continue
            status = str(item.get("status", "unknown")).lower().strip()
            if status not in {"occupied", "available", "unknown"}:
                status = "unknown"
            confidence = max(0, min(100, to_int(item.get("confidence"), 0)))
            evidence = str(item.get("evidence", ""))[:180]
            ports.append(
                {
                    "number": number,
                    "status": status,
                    "confidence": confidence,
                    "evidence": evidence,
                }
            )
            seen_numbers.add(number)
    else:
        occupied = data.get("occupied_ports", [])
        available = data.get("available_ports", [])
        for number in occupied if isinstance(occupied, list) else []:
            n = to_int(number)
            if n > 0 and n not in seen_numbers:
                ports.append({"number": n, "status": "occupied", "confidence": 0, "evidence": ""})
                seen_numbers.add(n)
        for number in available if isinstance(available, list) else []:
            n = to_int(number)
            if n > 0 and n not in seen_numbers:
                ports.append({"number": n, "status": "available", "confidence": 0, "evidence": ""})
                seen_numbers.add(n)

    ports.sort(key=lambda p: p["number"])
    total_ports = to_int(data.get("total_ports"), len(ports)) or len(ports)
    occupied_ports = [p["number"] for p in ports if p["status"] == "occupied"]
    available_ports = [p["number"] for p in ports if p["status"] == "available"]
    unknown_ports = [p["number"] for p in ports if p["status"] == "unknown"]

    image_quality = data.get("image_quality") if isinstance(data.get("image_quality"), dict) else {}
    quality_score = to_int(image_quality.get("score", data.get("confidence", 0)), 0)
    quality_flags = image_quality.get("flags", [])
    if not isinstance(quality_flags, list):
        quality_flags = []

    metrics_raw = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    visible_connectors = max(0, to_int(metrics_raw.get("visible_connectors"), 0))
    visible_cables = max(0, to_int(metrics_raw.get("visible_cables"), 0))
    visible_fibers = max(0, to_int(metrics_raw.get("visible_fibers"), 0))

    # Validacion cruzada deterministica: cada puerto ocupado deberia
    # corresponder a un cable o fibra visible. Si hay muchos mas ocupados
    # que evidencia fisica observable, es un posible falso positivo.
    cable_evidence = max(visible_cables, visible_fibers)
    warnings: list[str] = []
    occupancy_consistent = True
    if len(occupied_ports) > cable_evidence + OCCUPANCY_CABLE_TOLERANCE:
        occupancy_consistent = False
        warnings.append(
            f"Posible falso positivo: {len(occupied_ports)} puertos marcados como "
            f"ocupados pero solo {cable_evidence} cables/fibras visibles."
        )

    avg_confidence = (
        round(sum(p["confidence"] for p in ports) / len(ports), 1) if ports else 0
    )

    needs_review = bool(
        unknown_ports
        or quality_score < 70
        or len(ports) != total_ports
        or not occupancy_consistent
        or (ports and avg_confidence < 75)
    )
    mensaje = (
        f"Total puertos {total_ports}. "
        f"Puertos ocupados: {format_port_list(occupied_ports)}. "
        f"Puertos libres: {format_port_list(available_ports)}."
    )
    if unknown_ports:
        mensaje += f" Puertos dudosos: {format_port_list(unknown_ports)}."
    return {
        "total_puertos": total_ports,
        "puertos_ocupados": occupied_ports,
        "puertos_libres": available_ports,
        "puertos_dudosos": unknown_ports,
        "mensaje": mensaje,
        "total_ports": total_ports,
        "occupied_count": len(occupied_ports),
        "occupied_ports": occupied_ports,
        "available_count": len(available_ports),
        "available_ports": available_ports,
        "unknown_count": len(unknown_ports),
        "unknown_ports": unknown_ports,
        "ports": ports,
        "image_quality": {
            "score": max(0, min(100, quality_score)),
            "flags": [str(flag) for flag in quality_flags[:8]],
        },
        "metrics": {
            "visible_connectors": visible_connectors,
            "visible_cables": visible_cables,
            "visible_fibers": visible_fibers,
        },
        "validation": {
            "avg_confidence": avg_confidence,
            "occupancy_consistent": occupancy_consistent,
            "warnings": warnings,
        },
        "needs_review": needs_review,
        "summary": str(data.get("summary", ""))[:300],
        "image": image_meta,
        "_source": source,
    }


def format_port_list(ports: list[int]) -> str:
    return ",".join(str(port) for port in ports) if ports else "ninguno"


async def read_image_from_request(request: Request) -> tuple[bytes, str, str | None]:
    content_type = request.headers.get("content-type", "").lower()

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(422, "El campo multipart 'image' debe ser un archivo.")
        media_type = (upload.content_type or "image/jpeg").split(";")[0].lower()
        if media_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, "Formato de imagen no soportado.")
        image_bytes = await upload.read(MAX_IMAGE_BYTES + 1)
        return image_bytes, media_type, upload.filename

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(422, "Envia JSON {'image': 'https://...'} o multipart con campo image.")

    image_url = ImageInput(**payload).image
    image_bytes, media_type = await download_image(image_url)
    return image_bytes, media_type, image_url


@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": app.version,
        "learning": "active" if supabase_client else "inactive",
        "model": ANTHROPIC_MODEL,
        "verification": {
            "enabled": ENABLE_VERIFICATION,
            "model": VERIFICATION_MODEL if ENABLE_VERIFICATION else None,
        },
        "limits": {
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_concurrent_analyses": MAX_CONCURRENT_ANALYSES,
            "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        },
        "endpoints": {
            "health": "GET /api/health",
            "analyze_multipart": "POST /api/analyze con multipart/form-data campo image",
            "analyze_url": "POST /api/analyze con JSON {'image': 'https://url-imagen.jpg'}",
            "feedback": "POST /api/feedback con JSON {'pattern': '...'}",
            "patterns": "GET /api/patterns",
            "corrections": "GET /api/corrections",
        },
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "supabase_configured": bool(supabase_client),
    }


@app.post("/api/analyze")
async def analyze_image(request: Request):
    require_token(request)
    await enforce_rate_limit(request)

    image_bytes, _media_type, source_id = await read_image_from_request(request)
    processed_bytes, image_meta = preprocess_image(image_bytes)
    logger.info(
        "Analizando imagen source=%s hash=%s size=%s",
        source_id or "upload",
        image_meta["sha256"][:12],
        len(processed_bytes),
    )

    known = check_known_image(image_meta["sha256"])
    if known:
        return normalize_result(known, image_meta, source="memory")

    raw_text = await call_claude(processed_bytes, "image/jpeg")
    parsed = parse_json_response(raw_text)
    result = normalize_result(parsed, image_meta, source="claude")

    # Segunda pasada solo cuando el primer analisis quedo marcado para
    # revision (puertos dudosos, baja confianza o contradiccion de cables).
    # Evita duplicar costo en imagenes claras.
    if ENABLE_VERIFICATION and result["needs_review"]:
        logger.info("Disparando verificacion en segunda pasada model=%s", VERIFICATION_MODEL)
        try:
            verified = await verify_with_second_pass(processed_bytes, "image/jpeg", result)
            result = normalize_result(verified, image_meta, source="claude+verified")
        except HTTPException as exc:
            logger.warning("Verificacion no disponible (%s); se devuelve primer analisis.", exc.detail)
            result["validation"]["warnings"].append(
                "Segunda verificacion no disponible; se devuelve el primer analisis."
            )

    return result


@app.post("/api/feedback")
async def submit_feedback(data: FeedbackInput, request: Request):
    require_token(request)
    if not supabase_client:
        raise HTTPException(503, "Supabase no configurado.")
    pattern = data.pattern.strip()
    if len(pattern) < 12:
        raise HTTPException(400, "El patron es demasiado corto.")
    try:
        supabase_client.table("visual_patterns").insert(
            {"pattern": pattern, "active": True}
        ).execute()
        patterns_cache["expires_at"] = 0.0
        return {"status": "ok", "message": "Patron guardado."}
    except Exception as exc:
        logger.warning("Error guardando patron: %s", exc)
        raise HTTPException(500, "Error guardando patron.")


@app.get("/api/patterns")
async def list_patterns(request: Request):
    require_token(request)
    if not supabase_client:
        raise HTTPException(503, "Supabase no configurado.")
    try:
        result = (
            supabase_client.table("visual_patterns")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return {"total": len(result.data or []), "patterns": result.data or []}
    except Exception as exc:
        logger.warning("Error listando patrones: %s", exc)
        raise HTTPException(500, "Error listando patrones.")


@app.get("/api/corrections")
async def list_corrections(request: Request):
    require_token(request)
    if not supabase_client:
        raise HTTPException(503, "Supabase no configurado.")
    try:
        result = (
            supabase_client.table("port_corrections")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return {"total": len(result.data or []), "corrections": result.data or []}
    except Exception as exc:
        logger.warning("Error listando correcciones: %s", exc)
        raise HTTPException(500, "Error listando correcciones.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
