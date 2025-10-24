#!/usr/bin/env python3
import os, sys, math, re
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.src.ocr import extraer_tokens

def run():
    path = os.path.join(REPO_ROOT, 'data', 'ticket2.jpg')
    if not os.path.exists(path):
        print('ticket2.jpg not found')
        return 1
    tokens = extraer_tokens(path)
    # find total tokens
    total_tokens = [tk for tk in tokens if 'TOTAL' in tk.get('text','').upper()]
    print('Found', len(tokens), 'tokens, TOTAL tokens:', len(total_tokens))
    # build token_amounts similar to analysis: tokens that contain digits
    token_amounts = []
    for idx, tk in enumerate(tokens):
        txt = tk.get('text','')
        if not txt or not any(ch.isdigit() for ch in txt):
            continue
        conf = tk.get('conf') or 0
        token_amounts.append((idx, txt, conf, tk))

    if not token_amounts:
        print('No numeric tokens found')
        return 0

    centers = []
    for tt in total_tokens:
        centers.append(((tt.get('left',0) or 0)+(tt.get('width',0) or 0)/2, (tt.get('top',0) or 0)+(tt.get('height',0) or 0)/2))

    def score_item(item):
        idx, txt, conf, tk = item
        conf = conf or 0
        cx = (tk.get('left',0) or 0) + (tk.get('width',0) or 0)/2
        cy = (tk.get('top',0) or 0) + (tk.get('height',0) or 0)/2
        min_dist = 1e9
        for tx, ty in centers:
            d = math.hypot(cx - tx, cy - ty)
            if d < min_dist:
                min_dist = d
        digits = sum(1 for c in txt if c.isdigit())
        score = (conf * 3) + (digits * 12)
        if min_dist < 1e9:
            score += max(0, 150 - min_dist)
        if re.search(r"[\.,]\d{2}$", txt):
            score += 80
        return score

    scored = [(item, score_item(item)) for item in token_amounts]
    scored.sort(key=lambda x: x[1], reverse=True)
    print('\nTop numeric token candidates:')
    for (idx, txt, conf, tk), sc in scored[:20]:
        print(f" idx={idx:03d} txt={txt!r} conf={conf} left={tk.get('left')} top={tk.get('top')} w={tk.get('width')} h={tk.get('height')} score={sc:.1f}")
    return 0

if __name__ == '__main__':
    raise SystemExit(run())
