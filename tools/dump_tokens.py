import pytesseract, cv2
from pytesseract import Output
import os

# Ruta al ejecutable Tesseract en Windows (ajusta si tu instalación difiere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------------------------------------
# dump_tokens.py
# -------------------------------------------------------------
# Este script se usa para depurar la salida de Tesseract.
# Objetivo:
#  - Ejecutar pytesseract.image_to_data sobre una imagen de ticket.
#  - Listar tokens que contienen dígitos, ':' o '/' (hora/fechas/montos)
#  - Mostrar su confianza (conf) y coordenadas (left, top, width, height)
#
# Uso:
#  - Activa tu venv y corre:
#      & "venv\Scripts\python.exe" tools\dump_tokens.py
#  - Observa la salida en consola y identifica la posición de la hora/fecha/monto.
#  - Con esa información puedes recortar una ROI y aplicar un OCR más agresivo.
#
# Notas:
#  - El script prueba dos configuraciones de Tesseract: PSM=6 y PSM=7
#    y también una variante con whitelist (solo dígitos, '/', ':').
#  - Si tesseract no está en PATH, ajusta la variable tesseract_cmd arriba.
# -------------------------------------------------------------

ruta = os.path.join(os.path.dirname(__file__), '..', 'data', 'ticket2.png')
if not os.path.isfile(ruta):
    ruta = os.path.abspath(ruta)
print('Image path:', ruta)
img = cv2.imread(ruta)
print('Image loaded:', img is not None)

for psm in (6, 7):
    for cfg_extra in ('', ' -c tessedit_char_whitelist=0123456789/: -c preserve_interword_spaces=1'):
        cfg = f'--oem 3 --psm {psm}' + cfg_extra
        try:
            data = pytesseract.image_to_data(img, lang='spa', config=cfg, output_type=Output.DICT)
        except Exception as e:
            print('Error running tesseract', e)
            continue
        print('\n---- PSM', psm, 'cfg_extra:', cfg_extra, '----')
        n = len(data['text'])
        count = 0
        for i in range(n):
            t = data['text'][i].strip()
            if not t:
                continue
            if any(ch.isdigit() for ch in t) or ':' in t or '/' in t or '-' in t:
                print(i, 'text="%s" conf=%s left=%s top=%s w=%s h=%s' % (t, data['conf'][i], data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
                count += 1
            if count > 200:
                break
print('\nDone')
