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
    # Normalizar texto a mayúsculas para buscar keywords
    lines = [l.strip() for l in texto.split('\n') if l.strip()]
    txt_upper = '\n'.join(lines).upper()

    # Primero buscar líneas que contengan 'TOTAL' o 'SUBTOTAL' y extraer monto
    keywords = ['TOTAL', 'SUBTOTAL', 'TOTAL A PAGAR', 'IMPORTE', 'PAGO']
    pattern_num = r"\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})"
    for kw in keywords:
        for l in lines[::-1]:
            if kw in l.upper():
                m = re.search(pattern_num, l)
                if m:
                    return m.group()

    # Si no encuentra por keywords, buscar la última aparición de número con decimales
    all_nums = re.findall(pattern_num, txt_upper)
    if all_nums:
        return all_nums[-1]

    # Fallback al método simple
    return extraer_monto(texto)


def extraer_monto_por_boxes(raw_data):
    """Busca montos usando las coordenadas devueltas por image_to_data.
    Selecciona tokens numéricos en la zona inferior del ticket y el más a la derecha.
    """
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

    # Estimar altura del documento
    max_bottom = max(e['top'] + e['h'] for e in entries)
    height = max_bottom if max_bottom > 0 else max(e['top'] for e in entries)

    # Ajustar la zona inferior para incluir montos que estén algo más arriba
    threshold_top = height * 0.45
    bottom_entries = [e for e in entries if e['top'] >= threshold_top]
    if not bottom_entries:
        bottom_entries = entries

    # patrón robusto de montos: permitir números con o sin separador de miles y con decimales
    amount_re = re.compile(r"\d+(?:[\.,]\d{2})")

    # Preferir tokens que parezcan montos (tengan decimales)
    currency_tokens = [e for e in bottom_entries if amount_re.search(e['text'])]
    if currency_tokens:
        # elegir por mayor confianza y luego por posicion (right)
        currency_tokens.sort(key=lambda e: (e['conf'] or 0, e['right']))
        return currency_tokens[-1]['text']

    # fallback: buscar cualquier token numérico en bottom_entries, preferir por confianza
    nums = [e for e in bottom_entries if re.search(r"\d", e['text'])]
    if nums:
        # preferir tokens con confianza razonable y que parezcan montos (decimales o >=4 dígitos)
        def digit_count(s):
            return len(re.sub(r"[^0-9]", "", s))

        strong = [e for e in nums if (e['conf'] or 0) >= 45 and (amount_re.search(e['text']) or digit_count(e['text']) >= 4)]
        if strong:
            strong.sort(key=lambda e: (e['conf'] or 0, e['right']))
            return strong[-1]['text']

        # último recurso: devolver el numérico con mayor confianza
        nums.sort(key=lambda e: (e['conf'] or 0, e['right']))
        return nums[-1]['text']

    return None


def extraer_fecha_avanzada(raw_data, texto):
    """Busca fechas preferentemente en la parte superior del ticket usando raw_data;
    si falla, busca en el texto completo con varios patrones.
    """
    # patrones posibles (dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd, dd/mm/yy)
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
            # construir líneas por line_num si está disponible
            if top_entries:
                # juntar los textos de las top entries en orden de top/left
                top_text = ' '.join([e['text'] for e in sorted(top_entries, key=lambda x: (x['top'], x['i']))])
                for p in patterns:
                    m = re.search(p, top_text)
                    if m:
                        val = m.group()
                        # si es yy convertir a yyyy
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
    """Reconstruye líneas de texto a partir del dict devuelto por image_to_data.
    Intenta usar block_num/line_num/word_num si están; si no, agrupa por 'top' cercano.
    Devuelve una lista de strings ordenadas de arriba a abajo.
    """
    if not raw_data:
        return []

    texts = raw_data.get('text', [])
    n = len(texts)
    if n == 0:
        return []

    # Preferir line_num si está
    line_nums = raw_data.get('line_num')
    block_nums = raw_data.get('block_num')
    lefts = raw_data.get('left')
    tops = raw_data.get('top')

    lines = {}
    if line_nums is not None:
        for i in range(n):
            line_key = (int(block_nums[i]) if block_nums is not None else 0, int(line_nums[i]))
            lines.setdefault(line_key, []).append((int(lefts[i]) if lefts is not None else 0, str(texts[i]).strip()))
        # ordenar por block,line then by left
        ordered = []
        for k in sorted(lines.keys(), key=lambda x: (x[0], x[1])):
            row = ' '.join([w for _, w in sorted(lines[k], key=lambda x: x[0]) if w])
            ordered.append(row)
        return ordered

    # Si no hay line_num, agrupar por 'top' cercano
    entries = []
    for i in range(n):
        try:
            t = int(tops[i])
            l = int(lefts[i]) if lefts is not None else 0
        except Exception:
            t = 0
            l = 0
        entries.append((t, l, str(texts[i]).strip()))

    # cluster por top con tolerancia de 8 pixels
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
    """Busca montos en líneas reconstruidas. Recorre de abajo hacia arriba buscando keywords y números a la derecha."""
    if not lines:
        return None
    pattern_num = re.compile(r"\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})")
    keywords = ['TOTAL', 'SUBTOTAL', 'TOTAL A PAGAR', 'IMPORTE', 'PAGO']

    # buscar por keywords primero
    for l in reversed(lines):
        L = l.upper()
        for kw in keywords:
            if kw in L:
                m = pattern_num.search(l)
                if m:
                    return m.group()

    # si no hay keywords, buscar la última línea con número
    for l in reversed(lines):
        m = pattern_num.search(l)
        if m:
            return m.group()

    return None


def extraer_fecha_por_lines(lines):
    if not lines:
        return None
    # revisar las primeras 6 líneas (encabezado)
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
    # fallback: buscar en todo
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
    """Busca patrón de hora (HH:MM) y luego busca una fecha cercana a la derecha o en los siguientes tokens.
    Devuelve la fecha normalizada si se encuentra.
    """
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

    # ordenar por top (asc) then left (asc)
    entries.sort(key=lambda e: (e['top'], e['left']))

    time_re = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
    date_re1 = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")
    date_re2 = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{2}\b")

    # scan for time tokens
    for idx, e in enumerate(entries):
        if time_re.search(e['text']):
            # look rightwards in same 'line' (top within +/-8) for date in next few tokens
            top_ref = e['top']
            # collect next tokens with similar top or slightly below
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

    # fallback: buscar cualquier token que parezca fecha en todo el texto tokens
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
    """Recorta el header (zona superior-derecha), lo procesa agresivamente y corre Tesseract
    con whitelist de dígitos y separadores para localizar hora y fecha.
    """
    import cv2
    import pytesseract
    img = cv2.imread(ruta_imagen)
    if img is None:
        return None
    h, w = img.shape[:2]
    # zona superior 0..25% altura, derecha 60..100% ancho
    y1 = 0
    y2 = int(h * 0.25)
    x1 = int(w * 0.60)
    x2 = w
    roi = img[y1:y2, x1:x2]

    # mejora: escala grande, CLAHE, bilat, umbral
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    den = cv2.bilateralFilter(gray, 9, 75, 75)
    # aumentar contraste local
    _, th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # upscale
    th = cv2.resize(th, (int((x2-x1)*2.5), int((y2-y1)*2.5)), interpolation=cv2.INTER_CUBIC)

    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/: -c preserve_interword_spaces=1"
    try:
        text = pytesseract.image_to_string(th, lang='spa', config=cfg)
    except Exception:
        return None

    # buscar hora y fecha en el ROI text
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

    # si encontramos hora y fecha, retornamos fecha; si solo fecha también
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


def extraer_monto_cerca_total_improved(raw_data, top_tol=120, left_tol=0):
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

    confs = raw_data.get('conf', [])
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
                j_conf = int(float(confs[j])) if j < len(confs) else 0
            except Exception:
                continue
            if abs(j_top - t_top) <= top_tol:
                candidates.append({'i': j, 'text': s, 'left': j_left, 'top': j_top, 'conf': j_conf})

    if not candidates:
        return None

    # Prefer tokens that include decimals (e.g., '14499,00')
    amount_re = re.compile(r"\d+(?:[\.,]\d{2})")
    decimal_candidates = [c for c in candidates if amount_re.search(c['text'])]

    # obtener arrays auxiliares para alturas/anchos si están disponibles
    widths = raw_data.get('width', [])
    heights = raw_data.get('height', [])

    # calcular centros aproximados de tokens TOTAL
    total_centers = []
    for ti in total_idxs:
        try:
            t_left = int(lefts[ti])
        except Exception:
            t_left = 0
        try:
            t_top = int(tops[ti])
        except Exception:
            t_top = 0
        try:
            t_h = int(heights[ti]) if ti < len(heights) else 0
        except Exception:
            t_h = 0
        total_centers.append({'left': t_left, 'center_y': t_top + (t_h / 2.0) if t_h else t_top, 'h': t_h})

    def min_dist_to_total_point(cand_left, cand_center_y):
        if not total_centers:
            return abs(cand_left), float('inf')
        dists = []
        v_diffs = []
        for tc in total_centers:
            dists.append(abs(cand_left - tc['left']))
            v_diffs.append(abs(cand_center_y - tc['center_y']))
        return min(dists), min(v_diffs)

    def nearby_semantic_flag(cand):
        """Devuelve True si hay palabras tipo 'cambio' o 'efectivo' cerca (misma banda vertical)."""
        c_top = cand.get('top', 0)
        c_h = cand.get('h', 12) or 12
        for j, t in enumerate(texts):
            if not t:
                continue
            tj = str(t).lower()
            if any(k in tj for k in ('cambio', 'efectivo', 'camb', 'efec')):
                try:
                    jt = int(tops[j])
                except Exception:
                    jt = 0
                if abs(jt - c_top) <= max(12, c_h * 1.5):
                    return True
        return False

    if decimal_candidates:
        scored = []
        for c in decimal_candidates:
            # obtener left/top/conf/height
            cl = c.get('left', 0)
            ct = c.get('top', 0)
            ch = c.get('h', 0) if c.get('h') is not None else (int(heights[c['i']]) if c['i'] < len(heights) else 12)
            c_center_y = ct + (ch / 2.0)
            dist_x, dist_y = min_dist_to_total_point(cl, c_center_y)

            # base score: distancia horizontal menor es mejor
            score = float(dist_x)

            # si está alineado verticalmente con algún TOTAL (misma 'línea'), darle gran bonus
            aligned = dist_y <= max(12, max((tc['h'] or 12) for tc in total_centers) * 1.5) if total_centers else False
            if aligned:
                score -= 200.0
            else:
                # penalizar por diferencia vertical
                score += float(dist_y) * 1.5

            # penalizar si está cerca de palabras tipo 'cambio'/'efectivo'
            if nearby_semantic_flag(c):
                score += 150.0

            # preferir mayor confianza
            score -= (c.get('conf', 0) * 0.5)

            scored.append((score, -c.get('conf', 0), c))

        scored.sort(key=lambda x: (x[0], x[1]))
        return scored[0][2]['text']

    # Si no hay decimales, preferir tokens con >=4 dígitos (posibles montos sin decimales)
    def digit_count(s):
        return len(re.sub(r"[^0-9]", "", s))

    longnums = [c for c in candidates if digit_count(c['text']) >= 4]
    if longnums:
        # aplicar misma lógica de penalizaciones/bonos para longnums
        scored = []
        for c in longnums:
            cl = c.get('left', 0)
            ct = c.get('top', 0)
            ch = c.get('h', 0) if c.get('h') is not None else (int(heights[c['i']]) if c['i'] < len(heights) else 12)
            c_center_y = ct + (ch / 2.0)
            dist_x, dist_y = min_dist_to_total_point(cl, c_center_y)
            score = float(dist_x)
            aligned = dist_y <= max(12, max((tc['h'] or 12) for tc in total_centers) * 1.5) if total_centers else False
            if aligned:
                score -= 150.0
            else:
                score += float(dist_y) * 1.2
            if nearby_semantic_flag(c):
                score += 120.0
            score -= (c.get('conf', 0) * 0.3)
            scored.append((score, -c.get('conf', 0), c))
        scored.sort(key=lambda x: (x[0], x[1]))
        return scored[0][2]['text']

    # fallback: pick rightmost numeric token in the band, pero penalizando semanticamente
    numeric = [c for c in candidates if re.search(r"\d", c['text'])]
    if numeric:
        numeric.sort(key=lambda c: (nearby_semantic_flag(c), -c.get('conf', 0), c['left']))
        return numeric[-1]['text']

    return None

"""Funciones para extraer fecha, comercio, monto (vacío)
"""


def clasificar_gasto_simple(texto_clean: str = '', texto_lines=None, merchant: str = '', tokens=None):
    """Clasificador simple basado en reglas y keywords.

    Entrada:
      - texto_clean: texto ya limpiado (string)
      - texto_lines: lista de líneas (opcional)
      - merchant: nombre del comercio (opcional)
      - tokens: lista de tokens OCR (opcional)

    Salida: string con la categoría: 'ropa', 'comida', 'supermercado', 'transporte', 'salud', 'combustible', 'ocio', 'otros'
    """
    if texto_lines is None:
        texto_lines = []
    if tokens is None:
        tokens = []

    tx = (texto_clean or '').lower()
    ml = (merchant or '').lower()
    import difflib
    # intentar cargar configuración externa de categorías (Argentina, BA)
    import json
    import os
    import logging
    logger = logging.getLogger(__name__)

    # paths posibles, intentar ordenar desde la ruta del repo hacia arriba
    base_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(base_dir, '..', 'backend', 'configs', 'categories_ar_ba.json'),
        os.path.join(base_dir, '..', '..', 'backend', 'configs', 'categories_ar_ba.json'),
        os.path.join(base_dir, '..', '..', '..', 'backend', 'configs', 'categories_ar_ba.json'),
    ]
    cats = None
    cfg_path_used = None
    try:
        for p in candidates:
            p_norm = os.path.normpath(p)
            if os.path.exists(p_norm):
                cfg_path_used = p_norm
                break
        if cfg_path_used:
            with open(cfg_path_used, 'r', encoding='utf-8') as fh:
                raw = json.load(fh)
                cats = {}
                for catname, data_cat in raw.get('categorías', {}).items():
                    comercios = [c.strip().lower() for c in data_cat.get('comercios', [])]
                    palabras = [p.strip().lower() for p in data_cat.get('palabras_clave', [])]
                    cats[catname] = {'comercios': comercios, 'palabras_clave': palabras}
        else:
            logger.debug('categories_ar_ba.json not found in candidates: %s', candidates)
    except Exception as ex:
        logger.exception('Error loading categories config: %s', ex)
        cats = None

    # fallback integrado si no hay archivo
    if cats is None:
        CATS = {
            'ropa': ['ropa', 'zara', 'h&m', 'bershka', 'pull', 'mango', 'primark', 'stradivarius', 'forever', 'moda', 'boutique', 'tacon', 'botin'],
            'comida': ['restaurante', 'bar', 'caf', 'comida', 'burger', 'pizzeria', 'pizza', 'tapas', 'sushi', 'cafe', 'menu'],
            'supermercado': ['super', 'supermercado', 'mercadona', 'carrefour', 'dia', 'lidl', 'aldi', 'hiper', 'hipercor'],
            'transporte': ['taxi', 'uber', 'cabify', 'renfe', 'metro', 'autobus', 'bus', 'ave', 'billete', 'boleto'],
            'salud': ['farmacia', 'farm', 'medic', 'hospital', 'botica'],
            'combustible': ['gasolina', 'repsol', 'bp', 'cepsa', 'gasolinera', 'fuel'],
            'ocio': ['cine', 'teatro', 'entradas', 'ocio', 'museo', 'concierto']
        }
        cats = {k: {'comercios': [], 'palabras_clave': v} for k, v in CATS.items()}

    debug = {'loaded_from': cfg_path_used if 'cfg_path_used' in locals() else None, 'matched_by': None, 'method': None}

    # prioridad 1: merchant contiene nombre conocido (comparar tokens de merchant)
    for catname, data_cat in cats.items():
        for c in data_cat.get('comercios', []):
            if not c:
                continue
            # exact substring match
            if c in ml:
                debug['matched_by'] = c
                debug['method'] = 'substring'
                return catname, debug
            # fuzzy match: compare commerce name to merchant tokens and full merchant string
            try:
                # compare against merchant tokens
                merch_tokens = [t for t in re.split(r"\W+", ml) if t]
                for mt in merch_tokens:
                    r = difflib.SequenceMatcher(None, c, mt).ratio()
                    if r >= 0.75:
                        debug['matched_by'] = c
                        debug['method'] = f'fuzzy_token({mt})'
                        debug['score'] = r
                        return catname, debug
                # compare full merchant
                r2 = difflib.SequenceMatcher(None, c, ml).ratio()
                if r2 >= 0.7:
                    debug['matched_by'] = c
                    debug['method'] = 'fuzzy_full'
                    debug['score'] = r2
                    return catname, debug
            except Exception:
                pass

    # prioridad 2: texto limpio contiene keywords
    for catname, data_cat in cats.items():
        for k in data_cat.get('palabras_clave', []):
            if k and k in tx:
                return catname

    # prioridad 3: líneas (puede ayudar a detectar supermercados/restaurantes)
    for line in texto_lines:
        ll = line.lower()
        for catname, data_cat in cats.items():
            for k in data_cat.get('palabras_clave', []):
                if k and k in ll:
                    return catname

    # prioridad 4: tokens (buscar nombres de cadenas comunes)
    token_text = ' '.join([str(t.get('text','')).lower() if isinstance(t, dict) else str(t).lower() for t in tokens])
    for catname, data_cat in cats.items():
        for c in data_cat.get('comercios', []):
            if c and c in token_text:
                return catname

    # fallback
    return 'otros'


