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
    version="5.0.0-mvp",
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


BASE_PROMPT = """Eres un auditor experto en redes FTTH y nodos ATP.

Objetivo: detectar, contar y clasificar puertos opticos en la imagen.

Reglas visuales:
- OCCUPIED: conector SC/APC verde insertado en el puerto, especialmente si tiene cable/latiguillo de fibra saliendo hacia abajo o hacia atras.
- AVAILABLE: tapa protectora plana, plastico sin cable, o adaptador claramente vacio.
- UNKNOWN: puerto visible pero ambiguo por blur, sombra, oclusion, reflejo o angulo.
- No inventes puertos. Si no se ve con claridad, marca UNKNOWN.
- Si hay una sola fila, numera de izquierda a derecha empezando en 1.
- Si hay dos filas, numera primero la fila superior izquierda a derecha, luego la inferior izquierda a derecha.
- Cada puerto debe aparecer una sola vez.
- La suma de occupied + available + unknown debe ser igual a total_ports.
- En cajas ATP de 8 puertos, los numeros suelen estar impresos arriba del peine/adaptadores; usa esos numeros como referencia primaria.
- Si un puerto tiene una tapa blanca/gris cubriendo el adaptador y no hay conector ni cable, ese puerto esta AVAILABLE.
- Si un puerto tiene un cuerpo verde SC/APC ocupando el adaptador, ese puerto esta OCCUPIED aunque el cable salga fuera del encuadre.
- Ejemplo operativo 1: total_ports=8, puertos 1,2,3,4 con tapas o vacios son available; puertos 5,6,7,8 con conectores/cables verdes son occupied.
- Ejemplo operativo 2: total_ports=8, puerto 1 con conector verde insertado es occupied; puertos 2,3,4,5,6,7,8 con tapas blancas/verdes sin cable son available.

{patterns}

Evalua calidad de imagen con score 0-100 y flags entre:
blur, glare, occlusion, angle, low_resolution, cropped, dark.

Responde solo JSON valido, sin markdown, con este esquema exacto:
{{
  "total_ports": 0,
  "ports": [
    {{"number": 1, "status": "occupied", "confidence": 0, "evidence": "cable visible"}}
  ],
  "image_quality": {{"score": 0, "flags": []}},
  "summary": "breve resumen operativo"
}}"""


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


async def call_claude(image_bytes: bytes, media_type: str) -> str:
    if not anthropic_client:
        raise HTTPException(503, "ANTHROPIC_API_KEY no configurado.")

    prompt = build_prompt()
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    async with analysis_semaphore:
        try:
            logger.info(
                "Claude request model=%s media_type=%s image_size=%s",
                ANTHROPIC_MODEL,
                media_type,
                len(image_bytes)
            )

            message = await anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=900,
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

    needs_review = bool(unknown_ports or quality_score < 70 or len(ports) != total_ports)
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
    return normalize_result(parsed, image_meta, source="claude")


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
