"""DEPRECATED TOOL

This tool was used for local debugging and referenced specific ticket files.
It has been removed from the project to keep the repository free of
ticket-specific debugging scripts. If you need a token dump utility, please
use the OCR API (`backend/main.py`) or implement a new generic tool that
accepts an input image path.
"""
#
# Notas:
#  - El script prueba dos configuraciones de Tesseract: PSM=6 y PSM=7
#    y también una variante con whitelist (solo dígitos, '/', ':').
#  - Si tesseract no está en PATH, ajusta la variable tesseract_cmd arriba.
# -------------------------------------------------------------

# Editable: set IMAGE_PATH to the file you want to inspect (relative to repo or absolute)
IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ticket2.png')
ruta = IMAGE_PATH
if not os.path.isabs(ruta):
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
