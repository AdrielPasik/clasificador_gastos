Arrancar:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/main.py
```

Probar con curl:

```bash
curl -F "file=@/ruta/al/ticket.jpg" http://localhost:8000/api/ocr
```
