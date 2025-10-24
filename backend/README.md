# Backend OCR API

This folder contains a FastAPI-based OCR backend that exposes a single endpoint for extracting structured fields from receipt images.

Endpoint
--------
- POST /api/ocr
  - multipart/form-data with field `file` (image/jpeg or image/png)
  - Returns JSON with keys: `fecha` (ISO yyyy-mm-dd or null), `monto` (float or null), `texto` (string), optional `merchant` (string) and `monto_raw` (string)

Example response
----------------
{
  "fecha": null,
  "monto": 14499.0,
  "texto": "...raw OCR text...",
  "merchant": "ARCOS DORADOS",
  "monto_raw": "14499,00"
}

Run locally
-----------
1. Create and activate venv (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

2. Start the server:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

3. Test with curl (PowerShell):

```powershell
curl -F "file=@data/ticket2.jpg;type=image/jpeg" http://127.0.0.1:8000/api/ocr
```

Notes
-----
- CORS is configured to allow http://localhost:3000 by default.
- The analysis logic uses token-level OCR when available and scores candidates by confidence, digit-count and proximity to 'TOTAL' keywords. Heuristics are intentionally generic.

Tools
-----
You can find helper files in `backend/tools/`:
- `postman_collection_ocr.json` — Postman collection (v2.1) with a POST /api/ocr request.
- `curl_examples.txt` — curl and PowerShell snippets for testing.
# OCR Backend (FastAPI)

Pequeña API REST para procesar imágenes de tickets y extraer fecha, monto y texto crudo usando Tesseract + OpenCV.

Requisitos
- Python 3.10+
- Tener Tesseract OCR instalado (Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe` u otra ruta)

Instalación

1. Crear y activar un virtualenv:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Configurar opcionalmente `.env` basado en `.env.example` si necesitas apuntar a una ruta custom de tesseract.

Uso

Arrancar la API:

```powershell
python main.py
```

Endpoint principal
- POST http://localhost:8000/api/ocr
  - multipart/form-data con un campo `file` (tipo image/*)
  - Respuesta JSON de ejemplo:

```json
{
  "fecha": "2025-09-23",
  "monto": 14499.0,
  "texto": "...texto crudo extraído..."
}
```

Ejemplo `curl`:

```bash
curl -F "file=@/ruta/al/ticket.jpg" http://localhost:8000/api/ocr
```

Frontend (fetch) ejemplo:

```js
const fd = new FormData();
fd.append('file', fileInput.files[0]);
const res = await fetch('http://localhost:8000/api/ocr', { method: 'POST', body: fd });
const json = await res.json();
```

Notas
- CORS habilitado para http://localhost:3000
- Límite de tamaño aceptable: 10MB
- No se guardan archivos de prueba en el repo; todos los uploads van a `uploads/` con nombres únicos

