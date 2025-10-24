#!/usr/bin/env python3
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytesseract
from pytesseract import Output
import cv2


def find_path():
    p = os.path.join(REPO_ROOT, 'data', 'ticket2.jpg')
    if os.path.exists(p):
        return p
    p2 = os.path.join(REPO_ROOT, 'data', 'ticket2.png')
    if os.path.exists(p2):
        return p2
    return None


def run():
    path = find_path()
    if not path:
        print('ticket2 file not found under data/')
        return 1
    img = cv2.imread(path)
    if img is None:
        print('Could not read image:', path)
        return 2

    configs = [('--oem 3 --psm 6', 'PSM6'), ('--oem 3 --psm 11', 'PSM11'), ('--oem 3 --psm 3', 'PSM3')]
    for cfg, name in configs:
        print('\n----', name, cfg, '----')
        try:
            data = pytesseract.image_to_data(img, lang='spa', config=cfg, output_type=Output.DICT)
        except Exception as e:
            print('Tesseract error:', e)
            continue
        n = len(data['text'])
        for i in range(n):
            txt = data['text'][i].strip()
            conf = data['conf'][i]
            left = data['left'][i]
            top = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            if not txt:
                continue
            print(f"{i:03d} conf={conf:>3} left={left:>4} top={top:>4} w={w:>3} h={h:>3} text='{txt}'")

    return 0


if __name__ == '__main__':
    raise SystemExit(run())
