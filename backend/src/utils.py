import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / 'uploads'

# Ensure upload dir exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Environment helpers
def get_tesseract_cmd():
    from os import getenv
    cmd = getenv('TESSERACT_CMD')
    if cmd and cmd.strip():
        return cmd
    # fallback common Windows path
    default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default):
        return default
    return None

# Basic mime validation
def is_image_mimetype(mimetype: str) -> bool:
    if not mimetype:
        return False
    return mimetype.startswith('image/')

# safe path join for saved uploads
def unique_upload_path(filename: str) -> str:
    import uuid, time
    ext = os.path.splitext(filename)[1] or '.jpg'
    name = f"{int(time.time())}-{uuid.uuid4().hex}{ext}"
    return str(UPLOAD_DIR / name)
