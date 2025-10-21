import cv2
import pytesseract
import numpy as np
import os
import re

from .procesamiento import run_pipeline, preproc_alterno, preprocesar_ticket
from .procesamiento import ocr_multi_scale, roi_second_pass
from .analisis import (
    extraer_monto,
    extraer_fecha,
    extraer_monto_avanzado,
    extraer_monto_por_boxes,
    extraer_fecha_avanzada,
    reconstruct_lines_from_data,
    extraer_monto_por_lines,
    extraer_fecha_por_lines,
    extraer_fecha_por_tokens,
    scan_header_for_date,
)
from .utils import normalize_monto, normalize_date
from .utils import clean_text_for_amount, field_confidence

# --- Configuración de Tesseract ---
"""OCR utilities for clasificador_gastos

Este módulo expone la función principal `extraer_texto(ruta_imagen)` que aplica
varios pipelines de preprocesado y heurísticas basadas en `pytesseract` para
extraer el texto del ticket y detectar campos estructurados (monto, fecha).

Diseño:
- Funciones de preprocesado (CLAHE, denoise, deskew, resize) están en
  `preprocesar_ticket` y `preproc_alterno`.
- `run_pipeline` ejecuta pytesseract sobre una imagen preprocesada y devuelve
  texto filtrado por confianza y el raw_data.
- Varias funciones de extracción trabajan sobre `raw_data` (coords) o sobre
  texto limpio para localizar montos/fechas.

Nota: el archivo es deliberadamente completo — contiene heurísticas y
fallbacks que ayudan con tickets reales. Más adelante se puede refactorizar
en módulos (procesamiento/analisis/utils) sin cambiar la API pública.
"""

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def run_pipeline(img_proc, psm=6):
    """Run pytesseract on an already preprocessed image.

    Args:
        img_proc: image (numpy array) already preprocessed (thresholded/grayscaled)
        psm: Page segmentation mode to pass to Tesseract (default 6)

    Returns:
        dict with keys: text (filtered by confidence), mean_confidence, raw_data (image_to_data)
    """
    cfg = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(img_proc, lang='spa', config=cfg, output_type=pytesseract.Output.DICT)
    except Exception:
        text_raw = pytesseract.image_to_string(img_proc, lang='spa', config=cfg)
        return {'text': text_raw, 'mean_confidence': None, 'raw_data': None}

    words = []
    confs = []
    for i, w in enumerate(data.get('text', [])):
        word = w.strip()
        conf = data.get('conf', [])[i]
        try:
            conf_val = int(conf)
        except Exception:
            try:
                conf_val = int(float(conf))
            except Exception:
                conf_val = -1
        if word and conf_val > 25:
            words.append(word)
            confs.append(conf_val)

    text_join = ' '.join(words)
    if not text_join:
        text_join = pytesseract.image_to_string(img_proc, lang='spa', config=cfg)

    mean_conf = sum(confs) / len(confs) if confs else None
    text_join = re.sub(r"\s+", ' ', text_join).strip()
    return {'text': text_join, 'mean_confidence': mean_conf, 'raw_data': data}


def preproc_alterno(ruta):
    """Pipeline alternativo de preprocesado (más agresivo).

    Se diseñó para tickets con contraste bajo o ruido. Devuelve una imagen
    binarizada y reescalada adecuada para OCR.
    """
    img = cv2.imread(ruta)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    den = cv2.bilateralFilter(g, 9, 75, 75)
    _, th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = th.shape
    scale = max(1.0, 1800 / float(w))
    th = cv2.resize(th, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return th

def extraer_texto(ruta_imagen):
    # Verificar que la imagen exista
    if not os.path.isfile(ruta_imagen):
        raise FileNotFoundError(f"No se pudo encontrar la imagen: {ruta_imagen}")
    # Probamos múltiples pipelines y configuraciones, luego elegimos la salida con mayor confianza media
    # ahora usamos run_pipeline importado desde src.procesamiento

    # pipeline 1: el actual preprocesado
    p1 = preprocesar_ticket(ruta_imagen)
    # pipeline 2: invertido (por si el texto quedó blanco sobre negro)
    p2 = cv2.bitwise_not(p1)
    # pipeline 3: reprocesado con parámetros alternativos (más agresivo)
    p3 = preproc_alterno(ruta_imagen)

    # Ejecutar OCR multi-scale para obtener una referencia robusta
    try:
        ms_best = ocr_multi_scale(ruta_imagen, scales=(1.0, 1.5, 2.0), psm_list=(6,))
        raw_orig = ms_best.get('raw_data')
    except Exception:
        raw_orig = None

    candidates = []
    # probar combinaciones de psm
    for img_candidate in (p1, p2, p3):
        for psm in (6, 3, 4):
            candidates.append(run_pipeline(img_candidate, psm))

    # Elegir la mejor por mean_confidence
    best = max(candidates, key=lambda c: (c['mean_confidence'] or 0))

    texto_limpio = best['text']
    mean_conf = best['mean_confidence']
    # Forzar un image_to_data sobre la imagen preprocesada principal (p1)
    # y usarlo como raw_data de referencia para heurísticas basadas en posiciones.
    # Preferir raw_orig (OCR sobre imagen original) para heurísticas de posición
    if raw_orig:
        final_raw = raw_orig
    else:
        try:
            from pytesseract import Output
            final_raw = pytesseract.image_to_data(p1, lang='spa', config='--oem 3 --psm 6', output_type=Output.DICT)
        except Exception:
            final_raw = best.get('raw_data')


    # Priorizar heurísticas basadas en posición/lineas porque suelen ser más fiables
    monto = None
    if final_raw:
        # 1) boxes (zona inferior / derecha)
        try:
            monto_boxes = extraer_monto_por_boxes(final_raw)
            if monto_boxes:
                monto = monto_boxes
        except Exception:
            monto = None

        # 2) líneas reconstruidas a partir de final_raw
        if monto is None:
            try:
                lines_final = reconstruct_lines_from_data(final_raw)
                monto_lines = extraer_monto_por_lines(lines_final)
                if monto_lines:
                    monto = monto_lines
            except Exception:
                pass

        # 3) heurística alrededor de 'TOTAL' (mejorada)
        if monto is None:
            try:
                from .analisis import extraer_monto_cerca_total_improved
                m_total = extraer_monto_cerca_total_improved(final_raw)
                if m_total:
                    monto = m_total
            except Exception:
                pass

        # 4) token proximity (pares de tokens como '59' + '95')
        if monto is None:
            try:
                from .analisis import extraer_monto_por_tokens_proximity
                monto_tokprox = extraer_monto_por_tokens_proximity(final_raw)
                if monto_tokprox:
                    monto = monto_tokprox
            except Exception:
                pass

    # Si aún no, usar la heurística textual avanzada
    if monto is None:
        monto = extraer_monto_avanzado(texto_limpio)

    # ROI second-pass: recortar alrededor de TOTAL y re-OCR con whitelist
    if monto is None and final_raw:
        try:
            roi_m = roi_second_pass(ruta_imagen, final_raw)
            if roi_m:
                monto = roi_m
        except Exception:
            pass

    # Fecha: intentar heurística avanzada (priorizar zona superior usando final_raw)
    fecha = None
    if final_raw:
        fecha = extraer_fecha_avanzada(final_raw, texto_limpio)
    if fecha is None:
        fecha = extraer_fecha(texto_limpio)

    # Último recurso: reconstruir líneas desde raw_data y buscar monto/fecha en líneas
    if best.get('raw_data'):
        lines = reconstruct_lines_from_data(best.get('raw_data'))
        if monto is None:
            monto_lines = extraer_monto_por_lines(lines)
            if monto_lines:
                monto = monto_lines
        if fecha is None:
            fecha_lines = extraer_fecha_por_lines(lines)
            if fecha_lines:
                fecha = fecha_lines

    # Último recurso: buscar fecha por tokens (hora + fecha cercanos)
    if fecha is None and final_raw:
        fecha_tokens = extraer_fecha_por_tokens(final_raw)
        if fecha_tokens:
            fecha = fecha_tokens

    # Último recurso 2: escanear header ROI (probar varias regiones cercanas en caso que no esté en la esquina exacta)
    if fecha is None:
        # probar ejecutar image_to_data sobre la imagen original y sobre las variantes
        try:
            from pytesseract import Output
            img_orig = cv2.imread(ruta_imagen)
            raw_orig = pytesseract.image_to_data(img_orig, lang='spa', config='--oem 3 --psm 6', output_type=Output.DICT)
        except Exception:
            raw_orig = None

        # intentar sobre raw_orig
        if raw_orig:
            ft = extraer_fecha_por_tokens(raw_orig)
            if ft:
                fecha = ft

        # intentar sobre p2/p3 preprocesados
        if fecha is None:
            try:
                raw_p2 = pytesseract.image_to_data(p2, lang='spa', config='--oem 3 --psm 6', output_type=Output.DICT)
            except Exception:
                raw_p2 = None
            try:
                raw_p3 = pytesseract.image_to_data(p3, lang='spa', config='--oem 3 --psm 6', output_type=Output.DICT)
            except Exception:
                raw_p3 = None
            for rd in (raw_p2, raw_p3):
                if rd:
                    ft = extraer_fecha_por_tokens(rd)
                    if ft:
                        fecha = ft
                        break

        # último fallback: scan header ROI agresivo
        if fecha is None:
            fecha_header = scan_header_for_date(ruta_imagen)
            if fecha_header:
                fecha = fecha_header

    result = {
        'id_imagen': os.path.basename(ruta_imagen),
        'text_raw': texto_limpio,
        'mean_confidence': mean_conf,
        'fields': {
            'monto': {
                'raw': monto,
                'normalized': None,
            },
            'fecha': {
                'raw': fecha,
                'normalized': None,
            }
        },
        'raw_data': best.get('raw_data'),
    }

    # Normalizar monto y fecha cuando sea posible
    # Preparar token_conf si tenemos raw_data: intentar buscar token matching
    token_conf = None
    method_conf = 0.5
    try:
        fr = final_raw
        if fr and result['fields']['monto']['raw']:
            # buscar token exacto o parcial y tomar su conf
            txts = fr.get('text', [])
            confs = fr.get('conf', [])
            target = str(result['fields']['monto']['raw']).strip()
            # direct match
            for i, t in enumerate(txts):
                if str(t).strip() == target:
                    try:
                        token_conf = int(float(confs[i]))
                        break
                    except Exception:
                        token_conf = None
            # si no match directo, buscar token que contenga los dígitos
            if token_conf is None:
                pat = re.sub(r"[^0-9]", '', target)
                if pat:
                    for i, t in enumerate(txts):
                        if pat and pat in re.sub(r"[^0-9]", '', str(t)):
                            try:
                                token_conf = int(float(confs[i]))
                                break
                            except Exception:
                                token_conf = None
    except Exception:
        token_conf = None

    # limpiar texto del monto antes de normalizar
    try:
        cleaned = clean_text_for_amount(result['fields']['monto']['raw'])
        val, norm = normalize_monto(cleaned)
        result['fields']['monto']['normalized'] = norm
        result['fields']['monto']['value'] = val
    except Exception:
        result['fields']['monto']['normalized'] = None
        result['fields']['monto']['value'] = None

    # calcular confidence final del campo
    try:
        fc = field_confidence(method_conf=method_conf, token_conf=token_conf, mean_conf=result.get('mean_confidence'))
        result['fields']['monto']['confidence'] = fc
    except Exception:
        result['fields']['monto']['confidence'] = 0.0

    try:
        norm_date = normalize_date(result['fields']['fecha']['raw'])
        result['fields']['fecha']['normalized'] = norm_date
    except Exception:
        result['fields']['fecha']['normalized'] = None

    # confidence for fecha (simple heuristic)
    try:
        result['fields']['fecha']['confidence'] = field_confidence(method_conf=0.5, token_conf=None, mean_conf=result.get('mean_confidence'))
    except Exception:
        result['fields']['fecha']['confidence'] = 0.0

    return result

def preprocesar_ticket(ruta_imagen):
    # 1️⃣ Leer imagen
    img = cv2.imread(ruta_imagen)

    # 2️⃣ Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Mejorar contraste
    gray = cv2.equalizeHist(gray)

    # 4️⃣ Reducir ruido
    blur = cv2.medianBlur(gray, 3)
    blur = cv2.GaussianBlur(blur, (3,3), 0)

    # 5️⃣ Agudizar bordes
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(blur, -1, kernel)

    # 6️⃣ Umbral adaptativo y Otsu
    thresh1 = cv2.adaptiveThreshold(sharp, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    _, thresh2 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7️⃣ Combinar resultados y limpiar con morfología
    combined = cv2.bitwise_and(thresh1, thresh2)
    kernel_morph = np.ones((1,1), np.uint8)
    clean = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_morph)

    # 8️⃣ Deskew
    coords = np.column_stack(np.where(clean > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = clean.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(clean, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 9️⃣ Redimensionar
    scale_percent = 200
    width = int(rotated.shape[1] * scale_percent / 100)
    height = int(rotated.shape[0] * scale_percent / 100)
    resized = cv2.resize(rotated, (width, height), interpolation=cv2.INTER_LINEAR)

    return resized

# ----- Funciones para extraer datos clave -----
def extraer_monto(texto):
    # Detecta patrones de dinero: $12,345.67 o 12.345,67
    patrones = [r"\$\s?\d+[.,]?\d*", r"\d+[.,]?\d*\s?\$"]
    for p in patrones:
        match = re.search(p, texto)
        if match:
            return match.group()
    return None

def extraer_fecha(texto):
    # Detecta fechas en formato DD/MM/AAAA o DD-MM-AAAA
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

    # Considerar la zona inferior (último 40%)
    threshold_top = height * 0.6
    bottom_entries = [e for e in entries if e['top'] >= threshold_top]
    if not bottom_entries:
        bottom_entries = entries

    # patrón robusto de montos
    pattern_num = re.compile(r"\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})$")

    currency_tokens = [e for e in bottom_entries if pattern_num.search(e['text'])]
    if currency_tokens:
        # ordenar por posición (right) y preferir mayor confianza
        currency_tokens.sort(key=lambda e: (e['right'], e['conf']))
        return currency_tokens[-1]['text']

    # fallback: buscar cualquier token numérico en bottom_entries
    nums = [e for e in bottom_entries if re.search(r"\d", e['text'])]
    if nums:
        nums.sort(key=lambda e: (e['right'], e['conf']))
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

# ----- Main -----
if __name__ == "__main__":
    ruta = "./data/ticket2.png"
    try:
        resultado = extraer_texto(ruta)
        if isinstance(resultado, dict):
            print("Texto extraído:\n", resultado.get('text'))
            print("Mean confidence:", resultado.get('mean_confidence'))
            print("Monto detectado:", resultado.get('monto'))
            print("Fecha detectada:", resultado.get('fecha'))
        else:
            # backward compatibility
            texto, monto, fecha = resultado
            print("Texto extraído:\n", texto)
            print("Monto detectado:", monto)
            print("Fecha detectada:", fecha)
    except Exception as e:
        print("Error al extraer texto:", e)