import cv2
import os
import re
import numpy as np
import pytesseract
from pytesseract import Output
from .utils import get_tesseract_cmd

# Configure tesseract from env or fallback
tcmd = get_tesseract_cmd()
if tcmd:
    pytesseract.pytesseract.tesseract_cmd = tcmd


def _read_image(path: str):
    img = cv2.imread(path)
    return img


def _auto_scale(img):
    h, w = img.shape[:2]
    if w < 800:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return img


def _denoise(img):
    # usar bilateral o fastNl dependiendo del tamaño
    try:
        return cv2.bilateralFilter(img, 9, 75, 75)
    except Exception:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def _deskew(gray):
    # Estimar ángulo usando los pixeles de alto contraste y rotar
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] < 10:
        return gray
    try:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        # ajustar el ángulo devuelto por minAreaRect
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return gray


def _adjust_contrast(gray):
    # CLAHE on the L channel in LAB color space
    try:
        if len(gray.shape) == 2:
            # convert to BGR then LAB to reuse CLAHE
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            bgr = gray
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray_final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        return gray_final
    except Exception:
        # fallback simple histogram equalization
        try:
            return cv2.equalizeHist(gray)
        except Exception:
            return gray


def _morph_cleanup(th):
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # cerrar pequeños huecos
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        # eliminar ruido puntual
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        return opened
    except Exception:
        return th


def _binarize(gray):
    # elegir Otsu o adaptive según contraste estimado
    try:
        # estimar contraste simple
        low, high = np.percentile(gray, (2, 98))
        if (high - low) < 50:
            # bajo contraste -> adaptive
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 6)
        else:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    except Exception:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th


def _preprocess(path: str):
    img = _read_image(path)
    if img is None:
        raise ValueError('No se pudo leer la imagen')

    # escalar
    img = _auto_scale(img)
    # to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # deskew first (works on gray)
    gray = _deskew(gray)
    # denoise (color denoising expects 3 channels; apply to BGR copy)
    try:
        den = _denoise(img)
        gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = _denoise(gray)
    # enhance contrast
    gray = _adjust_contrast(gray)
    # binarize
    th = _binarize(gray)
    # morphological cleanup to reduce speckle and broken glyphs
    th = _morph_cleanup(th)
    return th


def _preprocess_aggressive(path: str):
    """A more aggressive preprocessing: stronger denoise, higher scaling and
    stronger morphological closing to try to join broken numbers."""
    img = _read_image(path)
    if img is None:
        raise ValueError('No se pudo leer la imagen')
    # upscale more
    h, w = img.shape[:2]
    scale = 3 if w < 1200 else 2
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    # stronger denoise
    try:
        gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    except Exception:
        pass
    gray = _adjust_contrast(gray)
    # adaptive binarize with smaller block size to preserve small glyphs
    try:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 4)
    except Exception:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # stronger closing
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    except Exception:
        pass
    return th


def extraer_texto(ruta_imagen: str) -> str:
    """Lee la imagen, aplica preprocesado y devuelve texto crudo mediante pytesseract."""
    img = _read_image(ruta_imagen)
    if img is None:
        raise ValueError('No se pudo leer la imagen')

    proc = _preprocess(ruta_imagen)
    cfg = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(proc, lang='spa', config=cfg)

    # Additional numeric-focused passes: whitelist digits and separators to improve
    # extraction of amounts (e.g. "355,37" or ".37" fragments). Run several
    # psm modes (6,7,11) because different layouts respond better to different modes.
    num_cfgs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.",
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,.",
        "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789,."
    ]
    num_texts = []
    for nc in num_cfgs:
        try:
            num_texts.append(pytesseract.image_to_string(proc, lang='spa', config=nc))
        except Exception:
            num_texts.append("")
    # join unique numeric-looking outputs
    num_text = "\n".join(t for t in num_texts if t)

    # If main text is empty, fallback to original image (try both normal and numeric passes)
    if not text or len(text.strip()) == 0:
        try:
            text = pytesseract.image_to_string(img, lang='spa', config=cfg)
        except Exception:
            text = ""
        try:
            # try same numeric configs on original image if preprocessed didn't help
            if not num_text or len(num_text.strip()) == 0:
                num_text2 = []
                for nc in num_cfgs:
                    try:
                        num_text2.append(pytesseract.image_to_string(img, lang='spa', config=nc))
                    except Exception:
                        num_text2.append("")
                num_text = "\n".join(t for t in num_text2 if t)
        except Exception:
            num_text = num_text or ""

    # Merge useful numeric-only lines from num_text into text when they contain
    # decimal-like patterns and the main text doesn't already have them.
    try:
        dec_in_text = bool(re.search(r"[0-9][\.,][0-9]{2}", text or ""))
        dec_in_num = bool(re.search(r"[0-9][\.,][0-9]{2}", num_text or ""))
        if dec_in_num and not dec_in_text:
            # append unique lines from num_text that contain digits
            lines_main = set(l.strip() for l in (text or "").splitlines() if l.strip())
            for ln in (num_text or "").splitlines():
                lns = ln.strip()
                if not lns: continue
                if lns in lines_main: continue
                if re.search(r"\d", lns):
                    text = (text or "") + "\n" + lns
    except Exception:
        pass

    return text


def extraer_tokens(ruta_imagen: str, cfg: str = '--oem 3 --psm 6', lang: str = 'spa'):
    """Devuelve lista de tokens detectados por pytesseract con sus confidencias y bbox.
    Cada token es dict { text, conf, left, top, width, height }.
    """
    img = _read_image(ruta_imagen)
    if img is None:
        raise ValueError('No se pudo leer la imagen')

    # Prefer running on the original image for token-level extraction because
    # preprocessing (binarization) can change token segmentation. If that fails,
    # fall back to the preprocessed image.
    try:
        data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=Output.DICT)
    except Exception:
        try:
            proc = _preprocess(ruta_imagen)
            data = pytesseract.image_to_data(proc, lang=lang, config=cfg, output_type=Output.DICT)
        except Exception as e:
            raise

    tokens = []
    n = len(data.get('text', []))
    for i in range(n):
        t = (data['text'][i] or '').strip()
        if not t:
            continue
        conf_raw = data.get('conf', [None] * n)[i]
        try:
            conf = int(float(conf_raw))
        except Exception:
            conf = None
        tokens.append({
            'text': t,
            'conf': conf,
            'left': int(data.get('left', [0] * n)[i]),
            'top': int(data.get('top', [0] * n)[i]),
            'width': int(data.get('width', [0] * n)[i]),
            'height': int(data.get('height', [0] * n)[i]),
        })
    # Second pass: try extracting numeric tokens from the preprocessed image with a whitelist
    try:
        proc = _preprocess(ruta_imagen)
        num_cfg = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,."
        data2 = pytesseract.image_to_data(proc, lang=lang, config=num_cfg, output_type=Output.DICT)
        n2 = len(data2.get('text', []))
        for i in range(n2):
            t = (data2['text'][i] or '').strip()
            if not t or not re.search(r"\d", t):
                continue
            conf_raw = data2.get('conf', [None] * n2)[i]
            try:
                conf = int(float(conf_raw))
            except Exception:
                conf = None
            token2 = {
                'text': t,
                'conf': conf,
                'left': int(data2.get('left', [0] * n2)[i]),
                'top': int(data2.get('top', [0] * n2)[i]),
                'width': int(data2.get('width', [0] * n2)[i]),
                'height': int(data2.get('height', [0] * n2)[i]),
            }
            # avoid duplicates: check if same text and bbox already present
            dup = False
            for ex in tokens:
                if ex.get('text') == token2['text'] and ex.get('left') == token2['left'] and ex.get('top') == token2['top']:
                    dup = True; break
            if not dup:
                tokens.append(token2)
    except Exception:
        # if preprocessed pass fails, keep original tokens
        pass

    # Third pass: aggressive preprocessed image with numeric whitelist and psm variants
    try:
        proc2 = _preprocess_aggressive(ruta_imagen)
        num_cfgs = [
            "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.",
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,.",
            "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789,."
        ]
        for nc in num_cfgs:
            try:
                data3 = pytesseract.image_to_data(proc2, lang=lang, config=nc, output_type=Output.DICT)
            except Exception:
                continue
            n3 = len(data3.get('text', []))
            for i in range(n3):
                t = (data3['text'][i] or '').strip()
                if not t or not re.search(r"\d", t):
                    continue
                conf_raw = data3.get('conf', [None] * n3)[i]
                try:
                    conf = int(float(conf_raw))
                except Exception:
                    conf = None
                token3 = {
                    'text': t,
                    'conf': conf,
                    'left': int(data3.get('left', [0] * n3)[i]),
                    'top': int(data3.get('top', [0] * n3)[i]),
                    'width': int(data3.get('width', [0] * n3)[i]),
                    'height': int(data3.get('height', [0] * n3)[i]),
                }
                dup = False
                for ex in tokens:
                    if ex.get('text') == token3['text'] and ex.get('left') == token3['left'] and ex.get('top') == token3['top']:
                        dup = True; break
                if not dup:
                    tokens.append(token3)
    except Exception:
        pass

    # Focused pass: look for the word "TOTAL" and run numeric OCR on a larger
    # crop around it. If we find a decimal fragment (like ".37") and a nearby
    # integer token (or adjacent integer tokens that can be concatenated to form
    # the integer part), build a merged token (e.g. "355,37") and append it to
    # the tokens list. This is conservative (only uses data within the TOTAL
    # crop) and helps recover fragmented totals.
    try:
        # use the regular preprocessed image to detect the TOTAL location
        proc = _preprocess(ruta_imagen)
        total_data = pytesseract.image_to_data(proc, lang=lang, config='--oem 3 --psm 6', output_type=Output.DICT)
        n_tot = len(total_data.get('text', []))
        total_indices = [i for i in range(n_tot) if (total_data.get('text', [])[i] or '').strip().upper() == 'TOTAL']
        if total_indices:
            # take the first occurrence (usually the printed total label)
            i = total_indices[0]
            t_left = int(total_data.get('left', [0]*n_tot)[i])
            t_top = int(total_data.get('top', [0]*n_tot)[i])
            t_w = int(total_data.get('width', [0]*n_tot)[i])
            t_h = int(total_data.get('height', [0]*n_tot)[i])

            h_img, w_img = img.shape[:2]
            # expand vertical pad to capture totals that may be far from the word TOTAL
            pad_v = 500
            crop_top = max(0, t_top - pad_v)
            crop_bottom = min(h_img, t_top + t_h + pad_v)
            # crop full width to capture right-aligned totals
            crop = img[crop_top:crop_bottom, 0:w_img]

            # run numeric passes on the crop
            crop_num_cfgs = [
                "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.",
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,.",
            ]
            # also try aggressive-preprocessed crop to recover faint digits
            try:
                # create an aggressive preprocessed version of the crop
                # convert crop to temporary image by writing and re-reading via OpenCV is avoided; apply local ops
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_gray = _deskew(crop_gray)
                crop_gray = _adjust_contrast(crop_gray)
                try:
                    crop_th = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 3)
                except Exception:
                    _, crop_th = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                crop_th = None
            crop_tokens = []
            for nc in crop_num_cfgs:
                try:
                    cd = pytesseract.image_to_data(crop, lang=lang, config=nc, output_type=Output.DICT)
                except Exception:
                    continue
                nn = len(cd.get('text', []))
                for j in range(nn):
                    tt = (cd['text'][j] or '').strip()
                    if not tt or not re.search(r"\d", tt):
                        continue
                    try:
                        conf_j = int(float(cd.get('conf', [None]*nn)[j]))
                    except Exception:
                        conf_j = None
                    token_crop = {
                        'text': tt,
                        'conf': conf_j,
                        'left': int(cd.get('left', [0]*nn)[j]),
                        'top': int(cd.get('top', [0]*nn)[j]),
                        'width': int(cd.get('width', [0]*nn)[j]),
                        'height': int(cd.get('height', [0]*nn)[j]),
                    }
                    # normalize top to original image coordinates
                    token_crop['top'] = token_crop['top'] + crop_top
                    crop_tokens.append(token_crop)

            if crop_tokens:
                # find decimal fragments like ".37" or ",37" and integer tokens
                decimals = [t for t in crop_tokens if re.fullmatch(r"[\.,]\d{2}", t['text'])]
                integers = [t for t in crop_tokens if re.fullmatch(r"\d+", t['text'])]

                # also consider tokens that already look like full amounts (e.g. 355,37)
                full_amounts = [t for t in crop_tokens if re.fullmatch(r"\d{1,3}[\.,]\d{2}", t['text'])]
                if full_amounts:
                    # add the highest-conf full amount if not already present
                    fa = sorted(full_amounts, key=lambda x: (x.get('conf') or 0), reverse=True)[0]
                    dup = any(ex.get('text') == fa['text'] and abs(ex.get('top',0)-fa.get('top',0))<5 for ex in tokens)
                    if not dup:
                        # adjust bbox relative to original
                        tok = fa.copy()
                        tokens.append(tok)
                elif decimals and integers:
                    # choose the best decimal (highest conf) and the nearest integer to its left
                    dec = sorted(decimals, key=lambda x: (x.get('conf') or 0), reverse=True)[0]
                    # group integer tokens by proximity on same baseline: concatenate adjacent ints if gap small
                    integers_sorted = sorted(integers, key=lambda x: x['left'])
                    merged_ints = []
                    if integers_sorted:
                        cur = integers_sorted[0].copy()
                        for it in integers_sorted[1:]:
                            gap = it['left'] - (cur['left'] + cur['width'])
                            # allow larger horizontal gaps for fragmented OCR and more vertical slack
                            if gap <= 40 and abs(it['top'] - cur['top']) <= 20:
                                # concatenate
                                cur['text'] = cur['text'] + it['text']
                                cur['width'] = (it['left'] + it['width']) - cur['left']
                                cur['conf'] = max(cur.get('conf') or 0, it.get('conf') or 0)
                            else:
                                merged_ints.append(cur)
                                cur = it.copy()
                        merged_ints.append(cur)

                    # find integer candidate to the left of decimal (prefer nearest)
                    cand = None
                    best_dist = None
                    for mi in merged_ints:
                        # require integer to be left of decimal (or overlap) to avoid pairing with unrelated right-aligned numbers
                        if mi['left'] <= dec['left']:
                            vert_dist = abs(mi['top'] - dec['top'])
                            horiz_dist = dec['left'] - (mi['left'] + mi['width'])
                            if horiz_dist < 0:
                                horiz_dist = 0
                            # allow larger vertical separation up to 600 px but penalize it
                            if vert_dist > 600:
                                continue
                            score_dist = vert_dist*0.8 + horiz_dist*0.5
                            if best_dist is None or score_dist < best_dist:
                                best_dist = score_dist
                                cand = mi

                    if cand:
                        # build merged text
                        int_part = cand['text']
                        dec_part = dec['text']
                        # normalize separators to comma
                        dec_part_norm = dec_part.replace('.', ',')
                        if dec_part_norm.startswith(','):
                            merged_text = f"{int_part}{dec_part_norm}"
                        else:
                            merged_text = dec_part_norm
                        # avoid duplicates
                        if not any(ex.get('text') == merged_text and abs(ex.get('top',0)-cand.get('top',0))<5 for ex in tokens):
                            merged_token = {
                                'text': merged_text,
                                'conf': max(cand.get('conf') or 0, dec.get('conf') or 0),
                                'left': cand['left'],
                                'top': cand['top'],
                                'width': (dec['left'] + dec['width']) - cand['left'],
                                'height': max(cand['height'], dec['height']),
                            }
                            tokens.append(merged_token)
    except Exception:
        # be conservative: fail silently and keep existing tokens
        pass

    # LAST RESORT: aggressive column-right sweep. This crops the rightmost
    # portion of the receipt, upscales heavily, applies aggressive preproc and
    # runs several numeric PSMs. Then it attempts a permissive merge between any
    # integer and decimal fragments found in that column, allowing larger
    # vertical separations (up to 1000 px) to match cases like this ticket.
    try:
        h_img, w_img = img.shape[:2]
        # crop rightmost 40% (covers right-aligned totals in many receipts)
        left_col = int(w_img * 0.6)
        crop_r = img[0:h_img, left_col:w_img]

        # upscale aggressively
        ch, cw = crop_r.shape[:2]
        scale = 4 if cw < 800 else 3
        crop_up = cv2.resize(crop_r, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)

        # aggressive grayscale/contrast/threshold
        try:
            crop_gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
            crop_gray = _deskew(crop_gray)
            crop_gray = _adjust_contrast(crop_gray)
            _, crop_th = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            crop_th = None

        right_cfgs = [
            "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.",
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,.",
            "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789,.",
            "--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789,.",
            "--oem 3 --psm 1 -c tessedit_char_whitelist=0123456789,."
        ]

        right_tokens = []
        for rc in right_cfgs:
            try:
                source_img = crop_th if crop_th is not None else crop_up
                rd = pytesseract.image_to_data(source_img, lang=lang, config=rc, output_type=Output.DICT)
            except Exception:
                continue
            rn = len(rd.get('text', []))
            for k in range(rn):
                tt = (rd['text'][k] or '').strip()
                if not tt or not re.search(r"\d", tt):
                    continue
                try:
                    conf_k = int(float(rd.get('conf', [None]*rn)[k]))
                except Exception:
                    conf_k = None
                tok = {
                    'text': tt,
                    'conf': conf_k,
                    'left': int(rd.get('left', [0]*rn)[k]) + left_col,  # map to original coords
                    'top': int(rd.get('top', [0]*rn)[k]),
                    'width': int(rd.get('width', [0]*rn)[k]),
                    'height': int(rd.get('height', [0]*rn)[k]),
                }
                # adjust top to original scale if upscaled
                if scale != 1:
                    tok['left'] = int(tok['left'] / scale)
                    tok['top'] = int(tok['top'] / scale)
                    tok['width'] = int(tok['width'] / scale) if tok['width'] else tok['width']
                    tok['height'] = int(tok['height'] / scale) if tok['height'] else tok['height']
                right_tokens.append(tok)

        # merge permissively: look for decimal fragments and integers in right_tokens
        decs = [t for t in right_tokens if re.search(r"[\.,]\d{2}$", t['text'])]
        ints_right = [t for t in right_tokens if re.fullmatch(r"\d{1,4}", t['text'])]
        ints_all = [t for t in tokens if re.fullmatch(r"\d{1,4}", t['text'])] + ints_right

        for dec in decs:
            # prefer integers located above or to the left; allow large vertical gap up to 1000 px
            candidates = []
            for itok in ints_all:
                if itok.get('left', 0) <= dec.get('left', 0):
                    vert = abs((itok.get('top') or 0) - (dec.get('top') or 0))
                    horiz = max(0, dec.get('left') - (itok.get('left') + (itok.get('width') or 0)))
                    if vert <= 1000:
                        candidates.append((vert, horiz, itok))
            if not candidates:
                continue
            # choose minimal (vert*0.8 + horiz*0.5)
            candidates = sorted(candidates, key=lambda x: (x[0]*0.8 + x[1]*0.5))
            best = candidates[0][2]
            merged_text = best['text'] + dec['text'].replace('.', ',') if dec['text'].startswith(('.',',')) else dec['text']
            # avoid duplicating identical merged tokens
            if not any(ex.get('text') == merged_text and abs(ex.get('top',0)-best.get('top',0))<10 for ex in tokens):
                merged_tok = {
                    'text': merged_text,
                    'conf': max(best.get('conf') or 0, dec.get('conf') or 0),
                    'left': best.get('left'),
                    'top': best.get('top'),
                    'width': (dec.get('left') + dec.get('width', 0)) - best.get('left'),
                    'height': max(best.get('height') or 0, dec.get('height') or 0)
                }
                tokens.append(merged_tok)
    except Exception:
        pass

    # Focused multi-scale sweep to the right of the 'TOTAL' token.
    # This is an extra aggressive attempt: crop from the TOTAL token to the
    # right edge, upscale at multiple scales, apply strong preproc and run
    # several numeric PSMs. Any full amount found (\d{1,3}[.,]\d{2}) is
    # added to tokens; if only fragments are found, attempt permissive merges.
    try:
        h_img, w_img = img.shape[:2]
        total_tokens = [t for t in tokens if (t.get('text') or '').strip().upper() == 'TOTAL']
        if total_tokens:
            total = total_tokens[0]
            # expand crop a bit to the left in case digits start slightly before TOTAL
            x1 = max(0, total.get('left', 0) - 80)
            # increase vertical padding to capture totals that may be printed a bit higher/lower
            y1 = max(0, total.get('top', 0) - 300)
            y2 = min(h_img, total.get('top', 0) + total.get('height', 0) + 300)
            x2 = w_img
            crop = img[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                scales = [2, 3, 4, 5, 6, 8]
                psm_list = [6, 7, 11, 3, 1]
                found_amounts = []
                found_decs = []
                for scale in scales:
                    try:
                        ch, cw = crop.shape[:2]
                        up = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
                        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                        gray = _deskew(gray)
                        gray = _adjust_contrast(gray)
                        # stronger closing/dilate to join broken digits
                        try:
                            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        except Exception:
                            th = gray
                        try:
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                            # for the largest scale be more aggressive joining broken digits
                            iters = 1 if scale < 8 else 2
                            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=iters)
                            th = cv2.dilate(th, kernel, iterations=iters)
                            # small extra closing for ultra-aggressive scale
                            if scale >= 8:
                                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2, iterations=1)
                        except Exception:
                            pass
                        # collect tokens for this scale into a local list so we can merge adjacent ints
                        local_tokens = []
                        for psm in psm_list:
                            cfgs = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789,."
                            try:
                                rd = pytesseract.image_to_data(th, lang=lang, config=cfgs, output_type=Output.DICT)
                            except Exception:
                                continue
                            rn = len(rd.get('text', []))
                            for k in range(rn):
                                tt = (rd['text'][k] or '').strip()
                                if not tt or not re.search(r"\d", tt):
                                    continue
                                try:
                                    conf_k = int(float(rd.get('conf', [None]*rn)[k]))
                                except Exception:
                                    conf_k = None
                                # map coords back to original image
                                left_k = int(rd.get('left', [0]*rn)[k] / scale) + x1
                                top_k = int(rd.get('top', [0]*rn)[k] / scale) + y1
                                width_k = int(rd.get('width', [0]*rn)[k] / scale) if rd.get('width', [0]*rn)[k] else 0
                                height_k = int(rd.get('height', [0]*rn)[k] / scale) if rd.get('height', [0]*rn)[k] else 0
                                token_k = {
                                    'text': tt,
                                    'conf': conf_k,
                                    'left': left_k,
                                    'top': top_k,
                                    'width': width_k,
                                    'height': height_k,
                                }
                                local_tokens.append(token_k)
                        # merge adjacent integer-only tokens from local_tokens to form multi-digit integers
                        try:
                            int_only = sorted([t for t in local_tokens if re.fullmatch(r"\d+", t['text'])], key=lambda x: (x['top'], x['left']))
                            merged_ints = []
                            if int_only:
                                cur = int_only[0].copy()
                                for it in int_only[1:]:
                                    gap = it['left'] - (cur['left'] + cur['width'])
                                    # allow larger gap and vertical slack to join broken digits produced at high scale
                                    if gap <= 120 and abs(it['top'] - cur['top']) <= 50:
                                        cur['text'] = cur['text'] + it['text']
                                        cur['width'] = (it['left'] + it['width']) - cur['left']
                                        cur['conf'] = max(cur.get('conf') or 0, it.get('conf') or 0)
                                    else:
                                        merged_ints.append(cur)
                                        cur = it.copy()
                                merged_ints.append(cur)
                            else:
                                merged_ints = []
                        except Exception:
                            merged_ints = []
                        # now add results: full amounts, decimal fragments and merged integers
                        for ttok in local_tokens:
                            tt = ttok['text']
                            top_k = ttok['top']
                            if re.fullmatch(r"\d{1,3}[\.,]\d{2}", tt):
                                if not any(ex.get('text') == tt and abs(ex.get('top',0)-top_k)<8 for ex in tokens):
                                    tokens.append(ttok)
                                    found_amounts.append(ttok)
                            elif re.fullmatch(r"[\.,]\d{2}", tt):
                                found_decs.append(ttok)
                        for mi in merged_ints:
                            if not any(ex.get('text') == mi['text'] and abs(ex.get('top',0)-mi.get('top',0))<8 for ex in tokens):
                                tokens.append(mi)
                                # merged ints are added as integer tokens — they may be merged later with decimals
                    except Exception:
                        continue

                # If we didn't find a full amount but found decimals + integers, permissively merge
                if not found_amounts and found_decs:
                    # prefer integers that are on the same baseline as TOTAL (within 30px)
                    integers_all = [t for t in tokens if re.fullmatch(r"\d+", t.get('text',''))]
                    integers = []
                    try:
                        total_top = total.get('top', 0) if total else None
                        if total_top is not None:
                            integers = [t for t in integers_all if abs((t.get('top',0) or 0) - total_top) <= 30]
                    except Exception:
                        integers = []
                    if not integers:
                        integers = integers_all
                    for dec in found_decs:
                        # find nearest integer to left allowing larger vertical gap
                        cand = None
                        best_score = None
                        for it in integers:
                            if it.get('left',0) <= dec.get('left',0):
                                vert = abs((it.get('top') or 0) - (dec.get('top') or 0))
                                horiz = dec.get('left') - (it.get('left') + (it.get('width') or 0))
                                if horiz < 0: horiz = 0
                                if vert > 1200:
                                    continue
                                score = vert*0.6 + horiz*0.4
                                if best_score is None or score < best_score:
                                    best_score = score; cand = it
                        if cand:
                            merged_text = cand['text'] + dec['text'].replace('.', ',') if dec['text'].startswith(('.',',')) else dec['text']
                            if not any(ex.get('text') == merged_text and abs(ex.get('top',0)-cand.get('top',0))<12 for ex in tokens):
                                merged_tok = {
                                    'text': merged_text,
                                    'conf': max(cand.get('conf') or 0, dec.get('conf') or 0),
                                    'left': cand.get('left'),
                                    'top': cand.get('top'),
                                    'width': (dec.get('left') + dec.get('width',0)) - cand.get('left'),
                                    'height': max(cand.get('height') or 0, dec.get('height') or 0),
                                }
                                tokens.append(merged_tok)
    except Exception:
        pass

    return tokens


# Bonus: devolver hipótesis con simple scoring
def extraer_hipotesis(texto: str):
    from .analisis import _find_amounts, _find_dates
    amounts = _find_amounts(texto)
    dates = _find_dates(texto)
    # score simple: preferir decimales
    def score_amount(a):
        s = 0
        if ',' in a or '.' in a:
            s += 5
        digits = len([c for c in a if c.isdigit()])
        s += min(digits, 10)
        return s
    amounts_sorted = sorted(amounts, key=score_amount, reverse=True)
    return {'amounts': amounts_sorted, 'dates': dates}
