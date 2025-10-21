import os
import sys
import json

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.ocr import extraer_texto
import glob
import pytest


def test_extraer_image_env_or_first():
    # Usa TEST_IMAGE si está definida, sino usa la primera imagen encontrada en data/
    test_env = os.environ.get('TEST_IMAGE')
    if test_env:
        if not os.path.isabs(test_env):
            ruta = os.path.join(proj_root, 'data', test_env)
        else:
            ruta = test_env
    else:
        # buscar la primera imagen en data/
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            lst = glob.glob(os.path.join(proj_root, 'data', ext))
            if lst:
                ruta = lst[0]
                break
        else:
            pytest.skip('No test images found in data/')

    resultado = extraer_texto(ruta)
    assert isinstance(resultado, dict)
    fields = resultado.get('fields', {})
    assert fields.get('monto', {}).get('raw') is not None
    # normalizado puede ser None si el OCR fue muy ruidoso, pero intentamos que no
    assert fields.get('monto', {}).get('normalized') is not None
