import shutil
import os
import logging
import traceback   # <-- ESTA LÍNEA ES LA QUE RESUELVE EL ERROR
from .utils import unique_upload_path, is_image_mimetype
from .ocr import extraer_texto
from .analisis import extraer_campos
import cv2
import pytesseract
import re
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
from typing import Any

router = APIRouter()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

logger = logging.getLogger(__name__)


@router.post('/ocr', status_code=200)
async def ocr_endpoint(file: UploadFile = File(...), debug_tokens: bool = False, save_crops: bool = False) -> Any:

    allowed_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
    file_ct = file.content_type
    filename = (file.filename or '').lower()
    if not is_image_mimetype(file_ct):
        if not any(filename.endswith(ext) for ext in allowed_exts):
            raise HTTPException(status_code=400, detail='El archivo enviado no es una imagen válida')

    try:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
    except Exception:
        size = None
    if size and size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail='Archivo demasiado grande. Máx 10MB')

    target = unique_upload_path(file.filename)

    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        logger.info('Saving uploaded file to %s (size=%s)', target, size)
        with open(target, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(status_code=500, detail='No se pudo guardar el archivo')
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    try:
        texto = extraer_texto(target)
        try:
            tokens = []
            from .ocr import extraer_tokens
            tokens = extraer_tokens(target)
        except Exception:
            tokens = []
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("❌ Error OCR procesando imagen: %s", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={'detail': 'Error interno al procesar la imagen'})

    try:
        try:
            campos = extraer_campos(texto, tokens=tokens)
        except TypeError:
            campos = extraer_campos(texto)
    except Exception as e:
        logger.error("❌ Error al analizar texto OCR: %s", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={'detail': 'Error interno al analizar el texto'})

    try:
        curm = (campos.get('merchant') or '').strip()
        need_aggressive = (not curm) or len([c for c in curm if c.isalpha()]) < 4 or ('COMERCIO NO RECONOCIDO' in curm.upper())
        if need_aggressive:
            import cv2 as _cv2
            import pytesseract as _pyt
            img = _cv2.imread(target)
            if img is not None:
                hh, ww = img.shape[:2]
                roi = img[0:max(1, int(hh * 0.25)), 0:ww]
                try:
                    roi_up = _cv2.resize(roi, (ww * 3, int(hh * 0.25 * 3)), interpolation=_cv2.INTER_CUBIC)
                    gray = _cv2.cvtColor(roi_up, _cv2.COLOR_BGR2GRAY)
                    clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    gray = clahe.apply(gray)
                    _, th = _cv2.threshold(gray, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
                    try:
                        hdr_txt = _pyt.image_to_string(th, lang='spa', config='--oem 3 --psm 11')
                    except Exception:
                        hdr_txt = _pyt.image_to_string(roi_up, lang='spa', config='--oem 3 --psm 11')
                    if hdr_txt:
                        hu = hdr_txt.upper()
                        if any(p in hu for p in ('MC DON', 'MCDON', 'MCDONAL', "MCD", 'MC-')):
                            campos['merchant'] = "McDonald's"
                            campos['category_hint'] = campos.get('category_hint') or 'comida'
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if not campos.get('fecha') or campos.get('fecha') == '0000-00-00':
            for t in tokens:
                txt = t.get('text') if isinstance(t, dict) else str(t)
                digits = re.sub(r'\D', '', txt)
                if not digits:
                    continue
                cand = None
                if len(digits) == 8:
                    d = digits
                    day = int(d[0:2]); month = int(d[2:4]); year = int(d[4:8])
                    cand = (year, month, day)
                elif len(digits) == 7:
                    d = digits
                    day = int(d[0:2]); month = int(d[2:4]); year = 2000 + int(d[4:7])
                    cand = (year, month, day)
                elif len(digits) == 6:
                    d = digits
                    day = int(d[0:2]); month = int(d[2:4]); year = 2000 + int(d[4:6])
                    cand = (year, month, day)
                if cand:
                    yr, mo, da = cand
                    try:
                        import datetime as _dt
                        _dt.date(yr, mo, da)
                        campos['fecha'] = f"{yr:04d}-{mo:02d}-{da:02d}"
                        campos.setdefault('fecha_debug', {})
                        campos['fecha_debug']['reconstructed_from_token'] = txt
                        break
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        txt_up = (campos.get('texto_clean') or '').upper()
        token_text = ' '.join([t.get('text','') for t in tokens]).upper() if tokens else ''
        m_patterns = ['MCDON', "MCD", "MC DON", "MCDONALD", "MCDONALDS", "MCDONALS"]
        found_mc = any(p in txt_up or p in token_text for p in m_patterns)
        if found_mc:
            campos['merchant'] = "McDonald's"
            campos['category_hint'] = campos.get('category_hint') or 'comida'
    except Exception:
        pass

    crop_paths = []
    crop_ocr = []
    if save_crops and tokens:
        try:
            candidates = []
            for tk in tokens:
                t = (tk.get('text') or '').strip()
                if not t:
                    continue
                if re.search(r'20\d{2}', t) or re.search(r'\d{4}[^\d]+\d{2,3}', t):
                    candidates.append(tk)
            if not candidates:
                candidates = [tk for tk in tokens if re.search(r'\d{2,4}', (tk.get('text') or ''))]

            if candidates:
                cand = sorted(candidates, key=lambda x: ((x.get('conf') or 0), - (x.get('top') or 0)), reverse=True)[0]
                try:
                    img = cv2.imread(target)
                    h_img, w_img = img.shape[:2]
                    left = max(0, (cand.get('left', 0) or 0) - 160)
                    top = max(0, (cand.get('top', 0) or 0) - 120)
                    right = min(w_img, (cand.get('left', 0) or 0) + (cand.get('width', 0) or 0) + 160)
                    bottom = min(h_img, (cand.get('top', 0) or 0) + (cand.get('height', 0) or 0) + 120)
                    crop = img[top:bottom, left:right]
                    if crop is not None and crop.size > 0:
                        crop_name = unique_upload_path('date-crop.jpg')
                        cv2.imwrite(crop_name, crop)
                        crop_paths.append(crop_name)
                        cfgs = ["--oem 3 --psm 6", "--oem 3 --psm 7", "--oem 3 --psm 11"]
                        for cfg in cfgs:
                            try:
                                txt = pytesseract.image_to_string(crop, lang='spa', config=cfg)
                                if txt and txt.strip():
                                    crop_ocr.append({'cfg': cfg, 'text': txt})
                            except Exception:
                                continue
                except Exception:
                    pass
        except Exception:
            pass

    try:
        import importlib
        scan_date = None
        try:
            mod = importlib.import_module('src.analisis')
            scan_date = getattr(mod, 'scan_header_for_date', None)
        except Exception:
            try:
                mod = importlib.import_module('backend.src.analisis')
                scan_date = getattr(mod, 'scan_header_for_date', None)
            except Exception:
                scan_date = None

        if scan_date:
            hdr_date = scan_date(target)
            if hdr_date:
                m = re.search(r"(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})", hdr_date)
                if m:
                    d, mo, y = m.groups()
                    yy = int(y)
                    if len(y) == 2:
                        yy = 2000 + yy if yy < 50 else 1900 + yy
                    try:
                        iso = f"{yy:04d}-{int(mo):02d}-{int(d):02d}"
                        campos['fecha'] = iso
                    except Exception:
                        pass
        try:
            from .ocr import extraer_header_text
            hdr_text = extraer_header_text(target)
            if hdr_text:
                try:
                    for ln in hdr_text.splitlines():
                        clean_ln = re.sub(r"[^A-Za-zÀ-ÿ\s]", "", ln).strip()
                        if sum(1 for c in clean_ln if c.isalpha()) >= 3:
                            curm = campos.get('merchant') or ''
                            if not curm or sum(1 for c in curm if c.isalpha()) < len(clean_ln) - 2:
                                campos['merchant'] = clean_ln
                                break
                except Exception:
                    pass
                try:
                    hdr_upper = hdr_text.upper()
                    ym = re.search(r"(\d{1,2})\D+(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)\D+(20\d{2})", hdr_upper)
                    if ym:
                        d_s, m_s, y_s = ym.groups()
                        months = {'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,'SEPTIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12}
                        mo_num = months.get(m_s, None)
                        if mo_num:
                            try:
                                campos['fecha'] = f"{int(y_s):04d}-{mo_num:02d}-{int(d_s):02d}"
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

    resp = {
        'fecha': campos.get('fecha'),
        'monto': campos.get('monto'),
        'texto': campos.get('texto'),
        'texto_clean': campos.get('texto_clean'),
        'texto_lines': campos.get('texto_lines')
    }
    if crop_paths:
        resp['saved_crops'] = crop_paths
    if crop_ocr:
        resp['saved_crops_ocr'] = crop_ocr
    if campos.get('merchant'):
        resp['merchant'] = campos.get('merchant')
    if campos.get('monto_raw'):
        resp['monto_raw'] = campos.get('monto_raw')
    if debug_tokens:
        resp['tokens'] = tokens

    classifier = None
    import traceback
    import importlib
    import_types = []
    try:
        mod = importlib.import_module('src.analisis')
        classifier = getattr(mod, 'clasificar_gasto_simple', None)
        import_types.append('src.analisis')
    except Exception as e_src:
        import_types.append(f'src.analisis_error:{e_src}')
        try:
            mod = importlib.import_module('backend.src.analisis')
            classifier = getattr(mod, 'clasificar_gasto_simple', None)
            import_types.append('backend.src.analisis')
        except Exception as e_back:
            import_types.append(f'backend.src.analisis_error:{e_back}')

    if not classifier:
        resp['category'] = 'otros'
        resp['category_debug'] = {'import_attempts': import_types}
    else:
        try:
            cat_res = classifier(campos.get('texto_clean', ''), texto_lines=campos.get('texto_lines'), merchant=campos.get('merchant', ''), tokens=tokens)
            if isinstance(cat_res, tuple) and len(cat_res) >= 1:
                category = cat_res[0]
                debug_info = cat_res[1] if len(cat_res) > 1 else None
            else:
                category = cat_res
                debug_info = None
            if not category:
                resp['category'] = 'otros'
                resp['category_debug'] = debug_info or 'classifier_returned_none'
            else:
                resp['category'] = category
                if debug_info:
                    resp['category_debug'] = debug_info
        except Exception:
            resp['category'] = 'otros'
            resp['category_debug'] = {'exception': traceback.format_exc().splitlines()[-1], 'import_attempts': import_types}

    try:
        if campos.get('category_hint') and resp.get('category') in (None, 'otros'):
            resp['category'] = campos.get('category_hint')
            resp['category_debug'] = resp.get('category_debug', {})
            if isinstance(resp['category_debug'], dict):
                resp['category_debug']['hint_override'] = True
            else:
                resp['category_debug'] = {'hint_override': True, 'prev': resp.get('category_debug')}
    except Exception:
        pass

    try:
        merchant_val = campos.get('merchant') or ''
        if merchant_val:
            try:
                mod_check = importlib.import_module('src.analisis')
                mch_res = None
                try:
                    mch_res = mod_check.clasificar_gasto_simple('', texto_lines=campos.get('texto_lines', []), merchant=merchant_val, tokens=tokens)
                except Exception:
                    mch_res = None
                recognized = False
                if isinstance(mch_res, tuple) and len(mch_res) >= 2:
                    dbg = mch_res[1]
                    if isinstance(dbg, dict) and dbg.get('matched_by'):
                        recognized = True
                if not recognized:
                    resp['merchant'] = 'Comercio no reconocido por la aplicación'
                    resp['merchant_debug'] = {'recognized': False, 'original': merchant_val}
                else:
                    resp['merchant'] = merchant_val
                    resp['merchant_debug'] = {'recognized': True}
            except Exception:
                resp['merchant'] = campos.get('merchant')
                resp['merchant_debug'] = {'recognized': None, 'error': 'lookup_failed'}
        else:
            resp['merchant'] = None
    except Exception:
        pass

    return resp
