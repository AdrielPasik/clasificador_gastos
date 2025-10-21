"""Utilidades de normalización para montos y fechas."""
import re
from datetime import datetime


def normalize_monto(raw: str):
    """Intentar convertir una cadena de monto OCR a float.

    Devuelve tuple (float_or_none, normalized_str) where normalized_str is a
    cleaned representation (e.g. '7100.0') or None.
    """
    if not raw:
        return None, None
    s = str(raw).strip()
    # eliminar símbolos de moneda (conservar espacios para detectar '59 95')
    s = re.sub(r"[€$£¥]", '', s)
    # manejar paréntesis y signos
    s = s.replace('(', '-').replace(')', '')
    # Normalizar espacios repetidos
    s = re.sub(r"\s+", ' ', s).strip()

    # detectar si tiene ambos '.' y ','
    if '.' in s and ',' in s:
        # determinar cuál es decimal separador por posición
        if s.rfind('.') < s.rfind(','):
            # formato 1.234,56 -> quitar puntos y cambiar coma por punto
            s2 = s.replace('.', '').replace(',', '.')
        else:
            # formato 1,234.56 -> quitar comas
            s2 = s.replace(',', '')
    elif ',' in s:
        # si hay una sola coma y 3 dígitos después, podría ser thousand sep
        parts = s.split(',')
        if len(parts) == 2 and len(parts[1]) == 3:
            s2 = s.replace(',', '')
        else:
            s2 = s.replace(',', '.')
    else:
        # intentar detectar espacio separado como "59 95" -> 59.95
        m_sp = re.search(r"^(\d{1,4})\s+(\d{2})$", s)
        if m_sp:
            s2 = f"{m_sp.group(1)}.{m_sp.group(2)}"
        else:
            # Si sólo hay un punto y la parte tras el punto tiene 3 dígitos,
            # es muy probable que sea separador de miles: '7.100' -> 7100
            if '.' in s and s.count('.') == 1:
                left, right = s.split('.')
                if right.isdigit() and len(right) == 3 and left.isdigit():
                    s2 = left + right
                else:
                    s2 = s
            else:
                s2 = s

    # remover cualquier carácter no numérico excepto '-' y '.'
    s2 = re.sub(r"[^0-9\-\.]", '', s2)
    if s2 in ('', '-', '.'):
        return None, None
    try:
        val = float(s2)
        return val, f"{val:.2f}"
    except Exception:
        return None, None


def normalize_date(raw: str):
    """Normaliza fechas a ISO YYYY-MM-DD si es posible.

    Acepta formatos como DD/MM/YYYY, DD-MM-YYYY, DD/MM/YY, YYYY-MM-DD.
    """
    if not raw:
        return None
    s = str(raw).strip()
    # reemplazar guiones por slashes para facilitar parseo
    s2 = s.replace('-', '/').strip()
    # patterns to try
    patterns = ['%d/%m/%Y', '%d/%m/%y', '%Y/%m/%d']
    for p in patterns:
        try:
            dt = datetime.strptime(s2, p)
            # si año con 2 dígitos, strptime lo convierte correctamente (1900s/2000s)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            continue
    # fallback: try to extract numbers
    m = re.search(r"(\d{2})[/-](\d{2})[/-](\d{2,4})", s)
    if m:
        d, mo, y = m.groups()
        yy = int(y)
        if len(y) == 2:
            yyyy = 2000 + yy if yy < 50 else 1900 + yy
        else:
            yyyy = yy
        try:
            dt = datetime(yyyy, int(mo), int(d))
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return None
    return None


# ---------------- Correcciones y confianza ----------------
CHAR_CORRECTIONS = {
    'O': '0',
    'o': '0',
    'I': '1',
    'l': '1',
    'S': '5',
    's': '5',
    'B': '8',
    ',': ',',
}

def clean_text_for_amount(s: str):
    """Aplica correcciones simples a una cadena que representa un monto.

    - Reemplaza caracteres típicos confundidos por OCR (O->0, I->1, etc.)
    - Normaliza espacios y comas/puntos redundantes.
    """
    if not s:
        return s
    t = str(s)
    # arreglar caracteres comunes
    t2 = []
    for ch in t:
        if ch in CHAR_CORRECTIONS:
            t2.append(CHAR_CORRECTIONS[ch])
        else:
            t2.append(ch)
    out = ''.join(t2)
    # compactar espacios
    out = re.sub(r"\s+", ' ', out).strip()
    # si aparece formato 'NN NN' muy probablemente es decimal
    out = re.sub(r"(\d)\s+(\d{2})\b", r"\1,\2", out)
    return out


def field_confidence(method_conf=None, token_conf=None, mean_conf=None):
    """Combina varias fuentes de confianza en un valor 0..1.

    - token_conf: 0..100 (confidence del token usado)
    - mean_conf: 0..100 (mean confidence global)
    - method_conf: heurística propia (0..1) para priorizar métodos robustos
    """
    # normalizar
    tc = (token_conf or 0) / 100.0
    mc = (mean_conf or 0) / 100.0
    meth = method_conf if method_conf is not None else 0.5
    # combinación ponderada: token 0.5, mean 0.3, method 0.2
    score = 0.5 * tc + 0.3 * mc + 0.2 * meth
    # clamp
    return max(0.0, min(1.0, float(score)))
