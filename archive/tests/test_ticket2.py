import os
from src.ocr import extraer_texto


def test_ticket2_has_fields():
    base = os.path.dirname(os.path.dirname(__file__))
    img = os.path.join(base, 'data', 'ticket2.jpg')
    assert os.path.exists(img), f"Imagen de prueba no encontrada: {img}"

    res = extraer_texto(img)
    # Estructura mínima esperada
    assert 'fields' in res, 'La salida debe contener la clave "fields"'
    fields = res['fields']
    assert 'monto' in fields, 'Debe extraer "monto" en fields'
    assert 'fecha' in fields, 'Debe extraer "fecha" en fields'

    # Los campos deben incluir raw y normalized
    for k in ('raw', 'normalized'):
        assert 'raw' in fields['monto'] and 'normalized' in fields['monto'], 'Monto necesita raw y normalized'
        assert 'raw' in fields['fecha'] and 'normalized' in fields['fecha'], 'Fecha necesita raw y normalized'
