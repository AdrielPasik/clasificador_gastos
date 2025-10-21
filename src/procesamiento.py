"""Procesamiento de imágenes para OCR.

Contiene pipelines de preprocesado y wrapper de ejecución de Tesseract.
Estas funciones se diseñaron para ser importadas y usadas por `src.ocr`.
"""
import cv2
import pytesseract
import re


def run_pipeline(img_proc, psm=6):
    """Run pytesseract on an already preprocessed image.

    Args:
        img_proc: image (numpy array) already preprocessed
        psm: Page segmentation mode

    Returns:
        dict with keys text, mean_confidence, raw_data
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
        word = str(w).strip()
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
    """Pipeline alternativo más agresivo para imágenes con poco contraste."""
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


def preprocesar_ticket(ruta_imagen):
    """Preprocesado principal del ticket (CLAHE, blur, sharpen, threshold, deskew, resize).

    Mantiene la compatibilidad con la versión anterior en `src/ocr.py`.
    """
    import numpy as np

    img = cv2.imread(ruta_imagen)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.medianBlur(gray, 3)
    blur = cv2.GaussianBlur(blur, (3,3), 0)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(blur, -1, kernel)
    thresh1 = cv2.adaptiveThreshold(sharp, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    _, thresh2 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_and(thresh1, thresh2)
    kernel_morph = np.ones((1,1), np.uint8)
    clean = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_morph)
    coords = np.column_stack(np.where(clean > 0))
    angle = 0
    try:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    except Exception:
        angle = 0
    (h, w) = clean.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(clean, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    scale_percent = 200
    width = int(rotated.shape[1] * scale_percent / 100)
    height = int(rotated.shape[0] * scale_percent / 100)
    resized = cv2.resize(rotated, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized


def ocr_multi_scale(ruta_imagen, scales=(1.0, 1.5, 2.0), psm_list=(6,)):
    """Ejecuta OCR sobre varias escalas de la imagen y devuelve la mejor salida.

    Retorna un dict compatible con run_pipeline: {'text','mean_confidence','raw_data'}
    """
    import cv2
    import pytesseract
    from pytesseract import Output

    img = cv2.imread(ruta_imagen)
    h, w = img.shape[:2]
    best = {'text': '', 'mean_confidence': 0, 'raw_data': None}
    for scale in scales:
        nh = int(h * scale)
        nw = int(w * scale)
        img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        for psm in psm_list:
            cfg = f"--oem 3 --psm {psm}"
            try:
                data = pytesseract.image_to_data(img_s, lang='spa', config=cfg, output_type=Output.DICT)
                # compute mean conf similar to run_pipeline
                words = []
                confs = []
                for i, wtext in enumerate(data.get('text', [])):
                    wt = str(wtext).strip()
                    try:
                        c = int(float(data.get('conf', [])[i]))
                    except Exception:
                        c = -1
                    if wt and c > 25:
                        words.append(wt)
                        confs.append(c)
                text_join = ' '.join(words)
                if not text_join:
                    text_join = pytesseract.image_to_string(img_s, lang='spa', config=cfg)
                mean_conf = sum(confs) / len(confs) if confs else 0
                if mean_conf and mean_conf > (best.get('mean_confidence') or 0):
                    best = {'text': text_join, 'mean_confidence': mean_conf, 'raw_data': data}
            except Exception:
                continue
    return best


def roi_second_pass(ruta_imagen, raw_data, expand_w=300, expand_h=120):
    """Detecta token 'Total' en raw_data y hace un OCR específico en la región derecha cercana.

    Devuelve texto detectado (por ejemplo el monto) o None.
    """
    import cv2
    import pytesseract
    from pytesseract import Output
    if not raw_data:
        return None
    texts = raw_data.get('text', [])
    lefts = raw_data.get('left', [])
    tops = raw_data.get('top', [])
    widths = raw_data.get('width', [])
    heights = raw_data.get('height', [])
    # buscar indices que contengan 'total'
    total_idxs = [i for i, t in enumerate(texts) if 'total' in str(t).lower()]
    if not total_idxs:
        return None
    img = cv2.imread(ruta_imagen)
    h_img, w_img = img.shape[:2]
    # evaluar cada 'total' y recortar a la derecha
    candidates = []
    for ti in total_idxs:
        try:
            l = int(lefts[ti])
            t = int(tops[ti])
            wi = int(widths[ti])
            he = int(heights[ti])
        except Exception:
            continue
        x1 = max(0, l - 20)
        y1 = max(0, t - expand_h//2)
        x2 = min(w_img, l + wi + expand_w)
        y2 = min(h_img, t + he + expand_h//2)
        roi = img[y1:y2, x1:x2]
        if roi is None or roi.size == 0:
            continue
        # upscale and run whitelist OCR
        roi_up = cv2.resize(roi, (roi.shape[1]*2, roi.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,.-"
        try:
            txt = pytesseract.image_to_string(roi_up, lang='spa', config=cfg)
            candidates.append(txt.strip())
        except Exception:
            continue
    # devolver la candidate más plausible (la que contenga dígitos y separador)
    best = None
    for c in candidates:
        if c and any(ch.isdigit() for ch in c):
            best = c
            break
    return best
