import pytest
from backend.src.analisis import extraer_campos


def test_amount_parsing_simple():
    text = "TOTAL: 1.234,56\nGracias"
    res = extraer_campos(text, tokens=None)
    assert res['monto'] == pytest.approx(1234.56)


def test_merchant_extraction_from_tokens():
    # simulate tokens where ARCOS DORADOS appears near top
    tokens = [
        {'text': 'ARCOS', 'conf': 80, 'left': 100, 'top': 20, 'width': 50, 'height': 12},
        {'text': 'DORADOS', 'conf': 85, 'left': 160, 'top': 22, 'width': 80, 'height': 12},
        {'text': 'RESPONSABLE', 'conf': 90, 'left': 120, 'top': 60, 'width': 120, 'height': 12},
    ]
    # messy OCR text but tokens provided
    text = "RESPONSABLE IVA\nARCOS DORADOS\nTOTAL: 14.499,00"
    res = extraer_campos(text, tokens=tokens)
    # merchant should prefer ARCOS DORADOS cluster
    assert 'ARCOS' in res.get('merchant', '')
    assert 'DORADOS' in res.get('merchant', '')
