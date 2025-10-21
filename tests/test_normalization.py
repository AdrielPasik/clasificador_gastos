import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.utils import normalize_monto, clean_text_for_amount


def test_normalize_variants():
    cases = {
        '59 95': 59.95,
        '59,95': 59.95,
        '1.234,56': 1234.56,
        '1,234.56': 1234.56,
        '$ 7.100': 7100.0,
        'O,99': 0.99,  # O -> 0
    }
    for raw, expected in cases.items():
        cleaned = clean_text_for_amount(raw)
        val, norm = normalize_monto(cleaned)
        assert val is not None
        assert abs(val - expected) < 0.01
