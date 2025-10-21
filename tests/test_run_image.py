import os
import glob
import pytest
from src.ocr import extraer_texto

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, 'data')

# Allow the tester to specify a single image via env var TEST_IMAGE
TEST_IMAGE = os.environ.get('TEST_IMAGE')


def images_to_test():
    if TEST_IMAGE:
        if not os.path.isabs(TEST_IMAGE):
            return [(os.path.basename(TEST_IMAGE), os.path.join(DATA_DIR, TEST_IMAGE))]
        else:
            return [(os.path.basename(TEST_IMAGE), TEST_IMAGE)]
    # otherwise return all images in data/
    imgs = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        imgs.extend(glob.glob(os.path.join(DATA_DIR, ext)))
    return [(os.path.basename(p), p) for p in imgs]


@pytest.mark.parametrize('name,path', images_to_test())
def test_image_pipeline_minimum(name, path):
    assert os.path.exists(path), f"Imagen no encontrada: {path}"
    res = extraer_texto(path)
    assert isinstance(res, dict), f"Resultado inválido para {name}"
    assert 'fields' in res, f"Sin fields en resultado para {name}"
    fields = res['fields']
    assert 'monto' in fields and 'fecha' in fields, f"Faltan monto/fecha en {name}"
    for fld in ('monto', 'fecha'):
        assert 'raw' in fields[fld] and 'normalized' in fields[fld], f"{fld} no tiene raw/normalized en {name}"
