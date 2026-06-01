const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const previewSection = document.getElementById('preview-section');
const uploadSection = document.getElementById('upload-section');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-img');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const errorMessage = document.getElementById('error-message');
const portsList = document.getElementById('ports-list');
const portsUl = document.getElementById('ports-ul');

const API_URL = window.API_URL || (
    window.location.protocol.startsWith('http')
        ? `${window.location.origin}/api/analyze`
        : 'http://localhost:8001/api/analyze'
);

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (event) => handleFile(event.target.files[0]));

dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');
    handleFile(event.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showError('Por favor sube un archivo de imagen valido.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        uploadSection.classList.add('hidden');
        previewSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        portsList.classList.add('hidden');
    };
    reader.readAsDataURL(file);

    analyzeBtn.onclick = () => analyzeImage(file);
}

removeBtn.addEventListener('click', () => {
    fileInput.value = '';
    imagePreview.src = '';
    previewSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    portsList.classList.add('hidden');
});

async function analyzeImage(file) {
    setLoading(true);

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(result.detail || `Error API: ${response.status}`);
        }

        renderResults(result);
    } catch (error) {
        console.error(error);
        showError(error.message || 'Error al conectar con la API.');
    } finally {
        setLoading(false);
    }
}

function renderResults(data) {
    if (data.error) {
        showError(data.message || data.error);
        return;
    }

    const occupied = data.puertos_ocupados || data.occupied_ports || [];
    const available = data.puertos_libres || data.available_ports || [];
    const unknown = data.puertos_dudosos || data.unknown_ports || [];

    document.getElementById('res-total').textContent = data.total_puertos ?? data.total_ports ?? '-';
    document.getElementById('res-occupied').textContent = occupied.length;
    document.getElementById('res-available').textContent = available.length;

    const quality = data.image_quality?.score;
    const reviewText = data.needs_review ? ' Requiere revision tecnica.' : '';
    const unknownText = unknown.length ? ` Dudosos: ${formatList(unknown)}.` : '';
    const baseMessage = data.mensaje || `Total puertos ${data.total_puertos ?? data.total_ports ?? '-'}. Ocupados: ${formatList(occupied)}. Libres: ${formatList(available)}.`;
    const qualityText = quality === undefined ? '' : ` Calidad: ${quality}%.`;
    document.getElementById('res-message').textContent = `${baseMessage}${unknownText}${qualityText}${reviewText}`;

    renderPorts(data.ports || []);
    resultsSection.classList.remove('hidden');
}

function renderPorts(ports) {
    portsUl.innerHTML = '';

    if (!ports.length) {
        portsList.classList.add('hidden');
        return;
    }

    for (const port of ports) {
        const item = document.createElement('li');
        const status = port.status || 'unknown';
        const confidence = Number.isFinite(port.confidence) ? `${port.confidence}%` : '-';
        item.className = `port-item ${status}`;
        item.textContent = `Puerto ${port.number}: ${labelStatus(status)} (${confidence})`;
        if (port.evidence) {
            item.title = port.evidence;
        }
        portsUl.appendChild(item);
    }

    portsList.classList.remove('hidden');
}

function labelStatus(status) {
    if (status === 'occupied') return 'ocupado';
    if (status === 'available') return 'disponible';
    return 'ambiguo';
}

function formatList(values) {
    return values.length ? values.join(',') : 'ninguno';
}

function setLoading(isLoading) {
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');

    analyzeBtn.disabled = isLoading;
    if (isLoading) {
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
    } else {
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    setTimeout(() => errorMessage.classList.add('hidden'), 6000);
}
