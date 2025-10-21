"""Heurísticas para extraer campos (monto, fecha) desde OCR raw_data y texto.

Contiene funciones que extraen montos y fechas a partir de texto plano o del dict
devuelto por `pytesseract.image_to_data`.
"""
import re


def extraer_monto(texto):
    patrones = [r"\$\s?\d+[.,]?\d*", r"\d+[.,]?\d*\s?\$"]
    for p in patrones:
        match = re.search(p, texto)
        if match:
            return match.group()
    return None


def extraer_fecha(texto):
    patrones = [r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"]
    for p in patrones:
        match = re.search(p, texto)
        if match:
            return match.group()
    return None


def extraer_monto_avanzado(texto):
    lines = [l.strip() for l in texto.split('\n') if l.strip()]
    txt_upper = '\n'.join(lines).upper()
    keywords = ['TOTAL', 'SUBTOTAL', 'TOTAL A PAGAR', 'IMPORTE', 'PAGO']
    # patrón robusto: admite separators de miles y decimales con , o .
    pattern_num = r"\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})"
    # patrón que captura tokens separados por espacio como "59 95"
    pattern_space_decimal = re.compile(r"(\d{1,4})\s+(\d{2})")
    for kw in keywords:
        for l in lines[::-1]:
            if kw in l.upper():
                m = re.search(pattern_num, l)
                if m:
                    return m.group()
                # intentar capturar casos donde el entero y decimales están separados por espacio
                ms = pattern_space_decimal.search(l)
                if ms:
                    return f"{ms.group(1)},{ms.group(2)}"
    all_nums = re.findall(pattern_num, txt_upper)
    # si no hay matches directos con pattern_num, buscar tokens tipo "59 95"
    if not all_nums:
        ms_all = pattern_space_decimal.findall(txt_upper)
        if ms_all:
            # tomar el último por defecto
            g = ms_all[-1]
            return f"{g[0]},{g[1]}"
    if all_nums:
        return all_nums[-1]
    return extraer_monto(texto)


def extraer_monto_por_boxes(raw_data):
    if not raw_data:
        return None
    texts = raw_data.get('text', [])
    if not texts:
        return None
    entries = []
    for i, t in enumerate(texts):
        txt = str(t).strip()
        if not txt:
            continue
        try:
            conf = int(float(raw_data.get('conf', [])[i]))
        except Exception:
            conf = -1
        try:
            left = int(raw_data.get('left', [])[i])
            top = int(raw_data.get('top', [])[i])
            w = int(raw_data.get('width', [])[i])
            h = int(raw_data.get('height', [])[i])
        except Exception:
            left = top = w = h = 0
        entries.append({'text': txt, 'conf': conf, 'left': left, 'top': top, 'w': w, 'h': h, 'right': left + w})
    if not entries:
        return None
    max_bottom = max(e['top'] + e['h'] for e in entries)
    height = max_bottom if max_bottom > 0 else max(e['top'] for e in entries)
    threshold_top = height * 0.6
    bottom_entries = [e for e in entries if e['top'] >= threshold_top]
    if not bottom_entries:
        bottom_entries = entries
    pattern_num = re.compile(r"\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})$")
    currency_tokens = [e for e in bottom_entries if pattern_num.search(e['text'])]
    if currency_tokens:
        currency_tokens.sort(key=lambda e: (e['right'], e['conf']))
        return currency_tokens[-1]['text']
    nums = [e for e in bottom_entries if re.search(r"\d", e['text'])]
    if nums:
        nums.sort(key=lambda e: (e['right'], e['conf']))
        return nums[-1]['text']
    return None


def extraer_fecha_avanzada(raw_data, texto):
    """Busca fechas preferentemente en la parte superior del ticket usando raw_data;
    si falla, busca en el texto completo con varios patrones.
    """
    patterns = [r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", r"\b\d{2}[/-]\d{2}[/-]\d{2}\b", r"\b\d{4}[/-]\d{2}[/-]\d{2}\b"]

    # Intentar por raw_data en zona superior (primer 25%)
    if raw_data:
        texts = raw_data.get('text', [])
        tops = raw_data.get('top', [])
        heights = raw_data.get('height', [])
        entries = []
        for i, t in enumerate(texts):
            txt = str(t).strip()
            if not txt:
                continue
            try:
                top = int(tops[i])
                h = int(heights[i])
            except Exception:
                top = 0
                h = 0
            entries.append({'text': txt, 'top': top, 'h': h, 'i': i})

        if entries:
            max_bottom = max(e['top'] + e['h'] for e in entries)
            threshold = max(1, int(max_bottom * 0.25))
            top_entries = [e for e in entries if e['top'] <= threshold]
            if top_entries:
                top_text = ' '.join([e['text'] for e in sorted(top_entries, key=lambda x: (x['top'], x['i']))])
                for p in patterns:
                    m = re.search(p, top_text)
                    if m:
                        val = m.group()
                        if re.match(r"\d{2}[/-]\d{2}[/-]\d{2}", val):
                            parts = re.split(r"[/-]", val)
                            yy = int(parts[2])
                            yyyy = 2000 + yy if yy < 50 else 1900 + yy
                            val = f"{parts[0]}/{parts[1]}/{yyyy}"
                        return val

    # fallback: buscar en todo el texto
    for p in patterns:
        m = re.search(p, texto)
        if m:
            val = m.group()
            if re.match(r"\d{2}[/-]\d{2}[/-]\d{2}", val):
                parts = re.split(r"[/-]", val)
                yy = int(parts[2])
                yyyy = 2000 + yy if yy < 50 else 1900 + yy
                val = f"{parts[0]}/{parts[1]}/{yyyy}"
            return val

    return None


def reconstruct_lines_from_data(raw_data):
    if not raw_data:
        return []
    texts = raw_data.get('text', [])
    n = len(texts)
    if n == 0:
        return []
    line_nums = raw_data.get('line_num')
    block_nums = raw_data.get('block_num')
    lefts = raw_data.get('left')
    tops = raw_data.get('top')
    lines = {}
    if line_nums is not None:
        for i in range(n):
            line_key = (int(block_nums[i]) if block_nums is not None else 0, int(line_nums[i]))
            lines.setdefault(line_key, []).append((int(lefts[i]) if lefts is not None else 0, str(texts[i]).strip()))
        ordered = []
        for k in sorted(lines.keys(), key=lambda x: (x[0], x[1])):
            row = ' '.join([w for _, w in sorted(lines[k], key=lambda x: x[0]) if w])
            ordered.append(row)
        return ordered
    entries = []
    for i in range(n):
        try:
            t = int(tops[i])
            l = int(lefts[i]) if lefts is not None else 0
        except Exception:
            t = 0
            l = 0
        entries.append((t, l, str(texts[i]).strip()))
    entries.sort(key=lambda x: (x[0], x[1]))
    grouped = []
    current_top = None
    current_group = []
    for t, l, txt in entries:
        if current_top is None:
            current_top = t
            current_group = [(l, txt)]
            continue
        if abs(t - current_top) <= 8:
            current_group.append((l, txt))
        else:
            grouped.append(current_group)
            current_group = [(l, txt)]
            current_top = t
    if current_group:
        grouped.append(current_group)
    ordered = []
    for g in grouped:
        row = ' '.join([w for _, w in sorted(g, key=lambda x: x[0]) if w])
        ordered.append(row)
    return ordered


def extraer_monto_por_lines(lines):
    if not lines:
        return None
    pattern_num = re.compile(r"\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})")
    pattern_space_decimal = re.compile(r"(\d{1,4})\s+(\d{2})")
    keywords = ['TOTAL', 'SUBTOTAL', 'TOTAL A PAGAR', 'IMPORTE', 'PAGO']

    # Primero buscar líneas con keywords (de abajo hacia arriba)
    for l in reversed(lines):
        L = l.upper()
        for kw in keywords:
            if kw in L:
                # preferir matches con pattern_num (e.g., 1.234,56)
                nums = list(pattern_num.finditer(l))
                if nums:
                    return nums[-1].group()
                # si no, intentar detectar espacio-separado "59 95"
                ms = pattern_space_decimal.findall(l)
                if ms:
                    g = ms[-1]
                    return f"{g[0]},{g[1]}"

    # Si no se encontraron por keywords, buscar cualquier número en las últimas líneas
    for l in reversed(lines):
        m = pattern_num.search(l)
        if m:
            return m.group()
        ms = pattern_space_decimal.search(l)
        if ms:
            return f"{ms.group(1)},{ms.group(2)}"

    return None


def extraer_fecha_por_lines(lines):
    if not lines:
        return None
    header = '\n'.join(lines[:6])
    patrones = [r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", r"\b\d{2}[/-]\d{2}[/-]\d{2}\b", r"\b\d{4}[/-]\d{2}[/-]\d{2}\b"]
    for p in patrones:
        m = re.search(p, header)
        if m:
            val = m.group()
            if re.match(r"\d{2}[/-]\d{2}[/-]\d{2}", val):
                parts = re.split(r"[/-]", val)
                yy = int(parts[2])
                yyyy = 2000 + yy if yy < 50 else 1900 + yy
                val = f"{parts[0]}/{parts[1]}/{yyyy}"
            return val
    for p in patrones:
        m = re.search(p, '\n'.join(lines))
        if m:
            val = m.group()
            if re.match(r"\d{2}[/-]\d{2}[/-]\d{2}", val):
                parts = re.split(r"[/-]", val)
                yy = int(parts[2])
                yyyy = 2000 + yy if yy < 50 else 1900 + yy
                val = f"{parts[0]}/{parts[1]}/{yyyy}"
            return val
    return None


def extraer_fecha_por_tokens(raw_data):
    if not raw_data:
        return None
    texts = raw_data.get('text', [])
    if not texts:
        return None
    tops = raw_data.get('top', [])
    lefts = raw_data.get('left', [])
    entries = []
    for i, t in enumerate(texts):
        txt = str(t).strip()
        if not txt:
            continue
        try:
            top = int(tops[i])
        except Exception:
            top = 0
        try:
            left = int(lefts[i])
        except Exception:
            left = 0
        entries.append({'i': i, 'text': txt, 'top': top, 'left': left})
    entries.sort(key=lambda e: (e['top'], e['left']))
    time_re = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
    date_re1 = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")
    date_re2 = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{2}\b")
    for idx, e in enumerate(entries):
        if time_re.search(e['text']):
            top_ref = e['top']
            candidates = []
            for j in range(idx+1, min(len(entries), idx+12)):
                ej = entries[j]
                if abs(ej['top'] - top_ref) <= 12 or ej['top'] >= top_ref:
                    candidates.append(ej['text'])
                else:
                    break
            joined = ' '.join(candidates)
            m = date_re1.search(joined) or date_re2.search(joined)
            if m:
                val = m.group()
                if date_re2.match(val):
                    parts = re.split(r"[/-]", val)
                    yy = int(parts[2])
                    yyyy = 2000 + yy if yy < 50 else 1900 + yy
                    val = f"{parts[0]}/{parts[1]}/{yyyy}"
                return val
    full = ' '.join([e['text'] for e in entries])
    m = date_re1.search(full) or date_re2.search(full)
    if m:
        val = m.group()
        if date_re2.match(val):
            parts = re.split(r"[/-]", val)
            yy = int(parts[2])
            yyyy = 2000 + yy if yy < 50 else 1900 + yy
            val = f"{parts[0]}/{parts[1]}/{yyyy}"
        return val
    return None


def scan_header_for_date(ruta_imagen):
    import cv2
    import pytesseract
    img = cv2.imread(ruta_imagen)
    if img is None:
        return None
    h, w = img.shape[:2]
    y1 = 0
    y2 = int(h * 0.25)
    x1 = int(w * 0.60)
    x2 = w
    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    den = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.resize(th, (int((x2-x1)*2.5), int((y2-y1)*2.5)), interpolation=cv2.INTER_CUBIC)
    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/: -c preserve_interword_spaces=1"
    try:
        text = pytesseract.image_to_string(th, lang='spa', config=cfg)
    except Exception:
        return None
    time_re = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
    date_re1 = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")
    date_re2 = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{2}\b")
    tmatch = time_re.search(text)
    dmatch = date_re1.search(text) or date_re2.search(text)
    date_val = None
    if dmatch:
        date_val = dmatch.group()
        if date_re2.match(date_val):
            parts = re.split(r"[/-]", date_val)
            yy = int(parts[2])
            yyyy = 2000 + yy if yy < 50 else 1900 + yy
            date_val = f"{parts[0]}/{parts[1]}/{yyyy}"
    if date_val:
        return date_val
    return None


def extraer_monto_por_tokens_proximity(raw_data, max_dx=400, max_dy=30):
    """Busca montos formados por tokens cercanos, por ejemplo '59' y '95' separados.

    Recorre tokens ordenados por posición y busca parejas donde el segundo token
    tenga 2 dígitos (decimales) y esté a la derecha y con top similar.
    """
    if not raw_data:
        return None
    texts = raw_data.get('text', [])
    if not texts:
        return None
    lefts = raw_data.get('left', [])
    tops = raw_data.get('top', [])
    confs = raw_data.get('conf', [])

    entries = []
    for i, t in enumerate(texts):
        txt = str(t).strip()
        if not txt:
            continue
        try:
            l = int(lefts[i])
        except Exception:
            l = 0
        try:
            top = int(tops[i])
        except Exception:
            top = 0
        try:
            conf = int(float(confs[i]))
        except Exception:
            conf = 0
        entries.append({'i': i, 'text': txt, 'left': l, 'top': top, 'conf': conf})

    # ordenar por top asc, left asc
    entries.sort(key=lambda e: (e['top'], e['left']))

    for idx, e in enumerate(entries):
        a = e['text']
        # candidate integer part: digits (1-4)
        if re.fullmatch(r"\d{1,4}", a):
            # buscar siguientes tokens dentro de ventana
            # buscar tokens en una ventana amplia (adelante y uno atrás)
            for j in range(max(0, idx-3), min(len(entries), idx+6)):
                if j == idx:
                    continue
                b = entries[j]
                dx = abs(b['left'] - e['left'])
                dy = abs(b['top'] - e['top'])
                if dx <= max_dx and dy <= max_dy and re.fullmatch(r"\d{2}", b['text']):
                    # combinación encontrada (usar coma como separador decimal)
                    # Asegurar que el token decimal esté razonablemente a la derecha
                    if b['left'] >= e['left'] - 40:
                        return f"{a},{b['text']}"
    return None


def extraer_monto_cerca_total(raw_data, max_top_diff=40, max_dx=800):
    """Busca el token numérico más cercano a cualquier token que contenga 'TOTAL'.

    Prioriza tokens a la derecha y con top similar.
    """
    if not raw_data:
        return None
    texts = raw_data.get('text', [])
    lefts = raw_data.get('left', [])
    tops = raw_data.get('top', [])
    confs = raw_data.get('conf', [])

    # recolectar indices de tokens
    total_indices = [i for i, t in enumerate(texts) if 'total' in str(t).lower()]
    if not total_indices:
        return None

    # candidate numeric pattern (with decimals) and simple integers
    num_re = re.compile(r"\d{1,3}(?:[\.,]\d{2,3})?$")

    best_candidate = None
    best_score = None
    for ti in total_indices:
        try:
            t_left = int(lefts[ti])
            t_top = int(tops[ti])
        except Exception:
            continue
        # scan all tokens and score them
        for j, txt in enumerate(texts):
            s = str(txt).strip()
            if not s:
                continue
            if not re.search(r"\d", s):
                continue
            try:
                j_left = int(lefts[j])
                j_top = int(tops[j])
            except Exception:
                continue
            top_diff = abs(j_top - t_top)
            dx = j_left - t_left
            # ignore tokens too far vertically
            if top_diff > max_top_diff:
                continue
            # compute base score: vertical distance weighted heavily
            score = top_diff * 10
            # prefer tokens to the right (small positive dx is good)
            if dx >= 0:
                score += dx
            else:
                # penalize left tokens
                score += abs(dx) + 300
            # prefer tokens with decimals (',', '.') and higher confidence
            is_decimal = bool(re.search(r"[\.,]\d{2}$", s)) or bool(re.fullmatch(r"\d{1,4}\s+\d{2}", s))
            try:
                conf = int(float(confs[j]))
            except Exception:
                conf = 0
            # reduce score by confidence (better conf -> lower score)
            score -= conf * 0.5

            if is_decimal or num_re.search(s):
                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = s

    if best_candidate:
        return best_candidate
    return None


def extraer_monto_cerca_total_improved(raw_data, top_tol=36, left_tol=0):
    """Heurística determinista:
    - Localiza tokens que contienen 'total' (case-insensitive).
    - Recolecta tokens cuya 'top' esté dentro de top_tol píxeles.
    - Entre esos tokens, selecciona el token numérico más a la derecha que parezca un monto
      (prioriza tokens que contienen ',' o '.' o pattern 'NN NN').
    - Si no hay a la derecha, busca el número más cercano horizontalmente.
    """
    if not raw_data:
        return None
    texts = raw_data.get('text', [])
    lefts = raw_data.get('left', [])
    tops = raw_data.get('top', [])

    total_idxs = [i for i, t in enumerate(texts) if 'total' in str(t).lower()]
    if not total_idxs:
        return None

    def looks_like_amount(s):
        s = str(s).strip()
        if not s or not re.search(r"\d", s):
            return False
        if re.search(r"[\.,]\d{2}$", s):
            return True
        if re.fullmatch(r"\d{1,4}\s+\d{2}", s):
            return True
        return False

    candidates = []
    for ti in total_idxs:
        try:
            t_top = int(tops[ti])
            t_left = int(lefts[ti])
        except Exception:
            continue
        # gather tokens within vertical tolerance
        for j, txt in enumerate(texts):
            s = str(txt).strip()
            if not s:
                continue
            try:
                j_top = int(tops[j])
                j_left = int(lefts[j])
            except Exception:
                continue
            if abs(j_top - t_top) <= top_tol:
                candidates.append({'i': j, 'text': s, 'left': j_left, 'top': j_top})

    if not candidates:
        return None

    # prefer rightmost candidate that looks like amount
    right_candidates = [c for c in candidates if looks_like_amount(c['text']) and c['left'] >= t_left]
    if right_candidates:
        right_candidates.sort(key=lambda c: c['left'], reverse=True)
        return right_candidates[0]['text']

    # if none to the right, pick rightmost that looks like amount regardless
    any_amounts = [c for c in candidates if looks_like_amount(c['text'])]
    if any_amounts:
        any_amounts.sort(key=lambda c: c['left'], reverse=True)
        return any_amounts[0]['text']

    # fallback: pick rightmost numeric token in the band
    numeric = [c for c in candidates if re.search(r"\d", c['text'])]
    if numeric:
        numeric.sort(key=lambda c: c['left'], reverse=True)
        return numeric[0]['text']

    return None
"""Funciones para extraer fecha, comercio, monto (vacío)
"""


