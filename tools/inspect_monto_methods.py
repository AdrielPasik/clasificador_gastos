import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import cv2
import pytesseract
from pytesseract import Output
from src import analisis
from src import procesamiento
from src import ocr


def main():
    IMAGE_PATH = os.path.join(proj_root, 'data', 'ticket1.jpg')
    ruta = IMAGE_PATH
    img = cv2.imread(ruta)
    raw = pytesseract.image_to_data(img, lang='spa', config='--oem 3 --psm 6', output_type=Output.DICT)
    print('Tokens count:', len(raw.get('text', [])))
    print('\n--- Tokens (i, text, conf, left, top) ---')
    for i, t in enumerate(raw.get('text', [])):
        txt = str(t).strip()
        if not txt:
            continue
        conf = raw.get('conf', [])[i]
        left = raw.get('left', [])[i]
        top = raw.get('top', [])[i]
        print(i, repr(txt), conf, left, top)

    print('\n--- extraer_monto_por_tokens_proximity ---')
    print(analisis.extraer_monto_por_tokens_proximity(raw))

    print('\n--- extraer_monto_por_boxes ---')
    print(analisis.extraer_monto_por_boxes(raw))

    print('\n--- reconstruct_lines and extraer_monto_por_lines ---')
    lines = analisis.reconstruct_lines_from_data(raw)
    for i, l in enumerate(lines[:40]):
        print(i, l)
    print('->', analisis.extraer_monto_por_lines(lines))

    print('\n--- extraer_monto_cerca_total_improved ---')
    print(analisis.extraer_monto_cerca_total_improved(raw))

    print('\n--- extraer_monto_avanzado on full text ---')
    # full text from image
    txt_full = pytesseract.image_to_string(img, lang='spa', config='--oem 3 --psm 6')
    print(ocr.extraer_monto_avanzado(txt_full))


if __name__ == '__main__':
    main()
