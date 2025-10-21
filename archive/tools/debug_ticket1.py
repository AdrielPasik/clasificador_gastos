"""Debug: muestra líneas reconstruidas y tokens cercanos a 'TOTAL' para ticket1."""
import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.ocr import extraer_texto
from src.analisis import reconstruct_lines_from_data


def main():
    # Edita IMAGE_PATH abajo para cambiar la imagen que quieres depurar
    IMAGE_PATH = os.path.join(proj_root, 'data', 'ticket1.jpg')
    ruta = IMAGE_PATH
    res = extraer_texto(ruta)
    raw = res.get('raw_data')
    print('Mean confidence:', res.get('mean_confidence'))
    if raw:
        lines = reconstruct_lines_from_data(raw)
        print('\n--- RECONSTRUCTED LINES ---')
        for i, ln in enumerate(lines[-40:]):
            print(f'{i:02d}:', ln)
        print('\n--- TOKENS (indices, text, conf, left, top) ---')
        texts = raw.get('text', [])
        confs = raw.get('conf', [])
        lefts = raw.get('left', [])
        tops = raw.get('top', [])
        for i, t in enumerate(texts):
            if not str(t).strip():
                continue
            if any(ch.isdigit() for ch in str(t)) or 'TOTAL' in str(t).upper():
                print(i, repr(str(t)), confs[i] if i < len(confs) else None, lefts[i] if i < len(lefts) else None, tops[i] if i < len(tops) else None)
    else:
        print('No raw_data available')


if __name__ == '__main__':
    main()
