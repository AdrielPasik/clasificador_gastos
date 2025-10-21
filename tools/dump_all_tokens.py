import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import pytesseract
import cv2
from pytesseract import Output

IMAGE_PATH = os.path.join(proj_root, 'data', 'ticket1.jpg')
ruta = IMAGE_PATH
if not os.path.isabs(ruta):
    ruta = os.path.abspath(ruta)
img = cv2.imread(ruta)
cfg = '--oem 3 --psm 6'
data = pytesseract.image_to_data(img, lang='spa', config=cfg, output_type=Output.DICT)
print('n tokens', len(data.get('text', [])))
for i, t in enumerate(data.get('text', [])):
    txt = str(t).strip()
    if not txt:
        continue
    conf = data.get('conf', [])[i]
    left = data.get('left', [])[i]
    top = data.get('top', [])[i]
    width = data.get('width', [])[i]
    height = data.get('height', [])[i]
    print(i, repr(txt), conf, left, top, width, height)
