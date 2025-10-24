from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
from typing import Any
import shutil
import os
import logging
from .utils import unique_upload_path, is_image_mimetype
from .ocr import extraer_texto
from .analisis import extraer_campos

router = APIRouter()

# limitar tamaño de subida a 10MB
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

logger = logging.getLogger(__name__)


@router.post('/ocr', status_code=200)
async def ocr_endpoint(file: UploadFile = File(...), debug_tokens: bool = False) -> Any:
    # validar mime: aceptar si content_type apunta a image/*, o si el filename tiene extensión de imagen
    allowed_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
    file_ct = file.content_type
    filename = (file.filename or '').lower()
    if not is_image_mimetype(file_ct):
        if not any(filename.endswith(ext) for ext in allowed_exts):
            raise HTTPException(status_code=400, detail='El archivo enviado no es una imagen válida')

    # limitar tamaño (comprobación simple aquí: 10MB)
    try:
        # mover al final para averiguar tamaño
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
    except Exception:
        size = None
    if size and size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail='Archivo demasiado grande. Máx 10MB')
    # guardar archivo en uploads/
    target = unique_upload_path(file.filename)
    try:
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
        # also get token-level data to help analysis heuristics
        try:
            tokens = []
            from .ocr import extraer_tokens
            tokens = extraer_tokens(target)
        except Exception:
            tokens = []
    except ValueError as e:
        # error al leer imagen
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # error interno
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={'detail': 'Error interno al procesar la imagen'})

    try:
        # allow analysis to optionally accept token-level input
        try:
            campos = extraer_campos(texto, tokens=tokens)
        except TypeError:
            campos = extraer_campos(texto)
    except Exception:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={'detail': 'Error interno al analizar el texto'})

    resp = {
        'fecha': campos.get('fecha'),
        'monto': campos.get('monto'),
        'texto': campos.get('texto'),
        'texto_clean': campos.get('texto_clean'),
        'texto_lines': campos.get('texto_lines')
    }
    # optional debug fields
    if campos.get('merchant'):
        resp['merchant'] = campos.get('merchant')
    if campos.get('monto_raw'):
        resp['monto_raw'] = campos.get('monto_raw')
    if debug_tokens:
        # include token-level OCR output for debugging heuristics
        resp['tokens'] = tokens

    # clasificación simple basada en reglas
    # Intentar importar el clasificador desde la versión principal (`src/analisis.py`) que contiene
    # las últimas mejoras; si falla, caer a la versión local dentro de backend/src.
    classifier = None
    import traceback
    import importlib
    import_types = []
    try:
        # prefer absolute import from repo-level `src` package
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
        # no se pudo importar el clasificador; no bloqueamos la respuesta pero informamos
        resp['category'] = 'otros'
        resp['category_debug'] = {'import_attempts': import_types}
    else:
        try:
            cat_res = classifier(campos.get('texto_clean', ''), texto_lines=campos.get('texto_lines'), merchant=campos.get('merchant', ''), tokens=tokens)
            # classifier may return a string or a tuple (category, debug)
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

    return resp
