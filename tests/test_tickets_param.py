import os
import glob
import pytest
from src.ocr import extraer_texto

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, 'data')

# OPTIONAL: si quieres comprobar valores concretos, añade aquí { 'ticket1.jpg': {'monto': 59.95} }
EXPECTED = {
    # 'ticket1.jpg': {'monto': 59.95},
}


def get_images():
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for f in glob.glob(os.path.join(DATA_DIR, ext)):
            yield os.path.basename(f), f


@pytest.mark.parametrize('name,path', list(get_images()))
def test_ticket_minimum_fields(name, path):
    """Test parametrizado que ejecuta extraer_texto sobre cada imagen en data/.
    Comprueba que la salida contiene 'fields' con 'monto' y 'fecha', y que ambos tienen 'raw' y 'normalized'.
    """
    res = extraer_texto(path)
    assert isinstance(res, dict), f"Resultado inválido para {name}"
    assert 'fields' in res, f"Sin fields en resultado para {name}"
    fields = res['fields']
    assert 'monto' in fields and 'fecha' in fields, f"Faltan monto/fecha en {name}"
    for fld in ('monto', 'fecha'):
        assert 'raw' in fields[fld] and 'normalized' in fields[fld], f"{fld} no tiene raw/normalized en {name}"

    # si hay expectativas concretas, comprobarlas
    if name in EXPECTED:
        exp = EXPECTED[name]
        if 'monto' in exp:
            assert abs(fields['monto'].get('value', 0) - exp['monto']) < 1e-3
