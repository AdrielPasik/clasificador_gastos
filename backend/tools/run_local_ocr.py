#!/usr/bin/env python3
import os
import json
import sys

# Ensure repo root is on sys.path so `backend` package is importable when running this script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from backend.src.ocr import extraer_texto
    from backend.src.analisis import extraer_campos
except Exception as e:
    print('Error importing backend modules:', e)
    sys.exit(2)


def find_ticket_path():
    # prefer repo/data/ticket2.jpg
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, '..', '..', 'data', 'ticket2.jpg'),
        os.path.join(here, '..', '..', 'data', 'ticket2.jpeg'),
        os.path.join(here, '..', '..', 'data', 'ticket2.png'),
        os.path.join(os.getcwd(), 'data', 'ticket2.jpg')
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.exists(p):
            return p
    return None


def main():
    path = find_ticket_path()
    if not path:
        print('Could not find data/ticket2.jpg in the repository. Please place the file at data/ticket2.jpg')
        sys.exit(1)

    print('Processing image:', path)
    try:
        texto = extraer_texto(path)
    except Exception as e:
        print('Error during OCR:', e)
        sys.exit(3)

    try:
        # try to get token-level info for better analysis
        from backend.src.ocr import extraer_tokens
        try:
            tokens = extraer_tokens(path)
        except Exception:
            tokens = None
        campos = extraer_campos(texto, tokens)
    except Exception as e:
        print('Error during analysis:', e)
        sys.exit(4)

    print('\n=== RESULT ===')
    print(json.dumps(campos, ensure_ascii=False, indent=2))

    # save raw text for inspection
    outdir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.abspath(os.path.join(outdir, 'ticket2_raw.txt'))
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(texto)
    print('Raw OCR text saved to', raw_path)
    # save tokens dump if available
    if 'tokens' in locals() and tokens:
        tok_path = os.path.abspath(os.path.join(outdir, 'ticket2_tokens.json'))
        with open(tok_path, 'w', encoding='utf-8') as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)
        print('Token dump saved to', tok_path)


if __name__ == '__main__':
    main()
