"""OCR analysis helpers: extract date, amount and merchant from OCR output.

Conservative, general-purpose heuristics used by the FastAPI OCR endpoint.
The functions are intentionally defensive: they return sensible defaults and
expose debug fields so the frontend can display tokens/crops when needed.
"""

from typing import Optional, Dict, Any, List
import re
import math
from collections import defaultdict


# Patterns
_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})\b")
_SIMPLE_AMOUNT_RE = re.compile(r"\b\d+[\.,]?\d*\b")
_KEYWORDS = ["TOTAL", "TOT", "IMPORTE", "PAGO", "MONTO"]
_DATE_RE_LIST = [
    re.compile(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b"),
    re.compile(r"\b(\d{2})[/-](\d{2})[/-](\d{2})\b"),
    re.compile(r"\b(\d{4})[/-](\d{2})[/-](\d{2})\b"),
]
# also allow noisy separators, but require plausible year
_DATE_RE_LIST.append(re.compile(r"\b(\d{1,2})\D+(\d{1,2})\D+((?:19|20)\d{2}|\d{2})\b"))


# Conservative tunables (raise these to be less aggressive)
_AGGRESSIVE_MAX_PAIR_DIST = 400
_AGGRESSIVE_MIN_DEC_CONF = 55
_OVERRIDE_MIN_CURR_DIST = 600
_OVERRIDE_MAX_RECON_DIST = 250


def _norm_num(s: str) -> str:
    only = re.sub(r"[^0-9,\.]", "", s)
    # common OCR ambiguity: thousands separator vs decimal
    if only.count(",") == 1 and only.count(".") >= 1:
        return only.replace(".", "").replace(",", ".")
    return only.replace(",", ".")


def _normalize_year(n: int) -> int:
    return 2000 + n if n < 50 else 1900 + n


def _parse_date_match(m: re.Match) -> Optional[str]:
    try:
        a, b, c = m.groups()
        # patterns can be yyyy-mm-dd or dd-mm-yy or dd-mm-yyyy
        if len(a) == 4:
            yyyy, mm, dd = int(a), int(b), int(c)
        else:
            dd, mm = int(a), int(b)
            yyyy = _normalize_year(int(c)) if len(c) == 2 else int(c)
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        return None
    return None


def _find_dates(text: str) -> List[str]:
    out: List[str] = []

    # 0) Prioritize explicit "fecha" label followed by slash-date (common and reliable)
    # Look for occurrences of the word 'fecha' and then a nearby pattern like dd/mm/yyyy or yyyy/mm/dd
    for fm in re.finditer(r"\bfecha\b", text, flags=re.IGNORECASE):
        start = fm.end()
        window = text[start:start + 120]  # look a bit after the label
        # find slash-separated patterns in the window
        for sm in re.finditer(r"(\d{1,4}/\d{1,2}/\d{1,4})", window):
            cand = sm.group(1)
            parts = cand.split('/')
            # require one of the parts to be 4-digit year
            if any(len(p) == 4 for p in parts):
                # try to coerce into ISO date
                try:
                    if len(parts[0]) == 4:
                        yyyy = int(parts[0]); mm = int(parts[1]); dd = int(parts[2])
                    else:
                        dd = int(parts[0]); mm = int(parts[1]); yyyy = int(parts[2]) if len(parts[2]) == 4 else _normalize_year(int(parts[2]))
                    if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2099:
                        out.append(f"{yyyy:04d}-{mm:02d}-{dd:02d}")
                        # prefer the first reasonable match after 'fecha'
                        return out
                except Exception:
                    continue

    # 1) generic regex list (existing tolerant patterns)
    for pat in _DATE_RE_LIST:
        for m in pat.finditer(text):
            v = _parse_date_match(m)
            if v:
                out.append(v)

    # 2) conservative digit-window: try to find ddmmyyyy inside the digit stream
    if not out:
        s = re.sub(r"\D", "", text)
        if len(s) >= 8:
            for i in range(0, len(s) - 7):
                chunk = s[i:i+8]
                dd = int(chunk[0:2]); mm = int(chunk[2:4]); yy = int(chunk[4:8])
                if 1 <= dd <= 31 and 1 <= mm <= 12 and 1900 <= yy <= 2099:
                    out.append(f"{yy:04d}-{mm:02d}-{dd:02d}")
                    break

    return out


def _find_amounts(text: str, tokens: List[Dict[str, Any]] = None):
    """Return (monto_val, monto_raw, monto_debug_list).

    Conservative strategy:
    1) prefer explicit decimal patterns in text
    2) prefer repeated decimal tokens
    3) cautiously reconstruct integer+decimal fragments when near TOTAL
    4) fallback to first decimal-like substring
    """
    monto_debug: List[Dict[str, Any]] = []

    # Build candidate pool from several sources (text regex, token decimals, simple substrings)
    candidates: Dict[str, Dict[str, Any]] = {}

    def add_candidate(raw: str, pos: Optional[int] = None, token_info: Optional[Dict[str, Any]] = None):
        norm = _norm_num(raw)
        if not norm:
            return
        entry = candidates.setdefault(norm, {"raws": set(), "count": 0, "confs": [], "positions": [], "token_positions": []})
        entry["raws"].add(raw)
        entry["count"] += 1
        if token_info:
            entry["confs"].append(token_info.get('conf') or 0)
            entry["token_positions"].append(token_info)
        if pos is not None:
            entry["positions"].append(pos)

    # 1) explicit strong matches from full text
    for m in _AMOUNT_RE.finditer(text):
        add_candidate(m.group(), pos=m.start())

    # 2) token-level decimals
    if tokens:
        for tk in tokens:
            t = (tk.get('text') or '').strip()
            if re.search(r"[\.,]\d{2}$", t):
                add_candidate(t, pos=None, token_info=tk)

    # 3) simple decimal-like substrings (less confident)
    for m in re.finditer(r"\d+[\.,]\d{2}", text):
        add_candidate(m.group(), pos=m.start())

    # If no candidates, quick exit
    if not candidates:
        return None, None, monto_debug

    # Prepare keyword (TOTAL/MONTO/PAGO) positions
    kw_positions_chars: List[int] = []
    for kw in _KEYWORDS:
        for mm in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
            kw_positions_chars.append(mm.start())

    kw_token_positions: List[Dict[str, Any]] = []
    if tokens:
        for tk in tokens:
            t = (tk.get('text') or '').upper()
            if any(k in t for k in _KEYWORDS):
                kw_token_positions.append(tk)

    # Score candidates
    scored: List[Dict[str, Any]] = []
    for norm, info in candidates.items():
        score = 0.0
        cnt = info.get('count', 0)
        score += cnt * 20  # reward repetition strongly

        # prefer candidates that appear near a keyword in chars
        min_char_dist = None
        for p in info.get('positions', []):
            for kpos in kw_positions_chars:
                d = abs(p - kpos)
                if min_char_dist is None or d < min_char_dist:
                    min_char_dist = d
        if min_char_dist is not None:
            # closer gets more points; within 100 chars is strong
            score += max(0, 40 - (min_char_dist / 3))

        # token-based keyword proximity (pixels) when token positions are available
        min_tok_dist = None
        for tok in info.get('token_positions', []):
            if kw_token_positions:
                for kwt in kw_token_positions:
                    try:
                        cx = (tok.get('left', 0) or 0) + (tok.get('width', 0) or 0) / 2
                        cy = (tok.get('top', 0) or 0) + (tok.get('height', 0) or 0) / 2
                        kx = (kwt.get('left', 0) or 0) + (kwt.get('width', 0) or 0) / 2
                        ky = (kwt.get('top', 0) or 0) + (kwt.get('height', 0) or 0) / 2
                        d = math.hypot(cx - kx, cy - ky)
                        if min_tok_dist is None or d < min_tok_dist:
                            min_tok_dist = d
                    except Exception:
                        continue
        if min_tok_dist is not None:
            score += max(0, 40 - (min_tok_dist / 10))

        # prefer right-most values (usually totals are on the right)
        right_pref = 0.0
        for tok in info.get('token_positions', []):
            try:
                cx = (tok.get('left', 0) or 0) + (tok.get('width', 0) or 0) / 2
                right_pref = max(right_pref, cx)
            except Exception:
                continue
        if right_pref:
            # normalize by 1000px; tweak factor conservatively
            score += min(15.0, right_pref / 1000.0 * 15.0)

        # small bonus for having a confident token
        avg_conf = 0
        if info.get('confs'):
            avg_conf = sum(info['confs']) / len(info['confs'])
            score += (avg_conf / 100.0) * 10.0

        scored.append({"norm": norm, "info": info, "score": score})

    # pick best candidate
    scored.sort(key=lambda x: x['score'], reverse=True)
    best = scored[0]

    # require a reasonable score to accept; otherwise fallback to safer heuristics
    if best['score'] >= 25:
        chosen_norm = best['norm']
        info = best['info']
        raw_example = next(iter(info['raws']))
        try:
            val = float(chosen_norm)
            monto_debug.append({"chosen": raw_example, "parsed": val, "method": "scored_selection", "score": best['score']})
            return val, raw_example, monto_debug
        except Exception:
            pass

    # fallback to previous conservative approaches if scoring not decisive
    # token-level repeated decimals (as before)
    if tokens:
        dec_counts = defaultdict(int)
        dec_examples = {}
        for tk in tokens:
            t = (tk.get('text') or '').strip()
            if re.search(r"[\.,]\d{2}$", t):
                norm = _norm_num(t)
                dec_counts[norm] += 1
                if norm not in dec_examples or (tk.get('conf') or 0) > (dec_examples[norm].get('conf') or 0):
                    dec_examples[norm] = tk
        if dec_counts:
            best_norm, cnt = max(dec_counts.items(), key=lambda x: (x[1], x[0]))
            if cnt >= 2:
                try:
                    val = float(best_norm)
                    ex = dec_examples.get(best_norm)
                    raw = ex.get('text') if ex else best_norm
                    monto_debug.append({"chosen": raw, "parsed": val, "method": "token_most_common_decimal_fallback"})
                    return val, raw, monto_debug
                except Exception:
                    pass

    # final fallback: first explicit decimal-like substring
    for a in re.finditer(r"\d+[\.,]\d{2}", text):
        try:
            norm = _norm_num(a.group())
            return float(norm), a.group(), [{"chosen": a.group(), "parsed": float(norm), "method": "fallback_decimal_final"}]
        except Exception:
            continue

    return None, None, monto_debug


_PRODUCT_KEYWORDS = [
    'MEDIALUN', 'MEDIALUNA', 'CAPU', 'CAPPU', 'CAFE', 'CAFÉ', 'EMPANADA', 'SANDW', 'TORTA', 'PAN', 'BAGEL', 'MUFFIN'
]


def extraer_campos(texto: str, tokens: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not texto:
        return {"fecha": None, "monto": None, "texto": ""}

    original_text = texto
    texto_lines = [ln.rstrip() for ln in original_text.splitlines() if ln.strip()]
    texto_clean = "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in texto_lines)

    # Fecha
    fecha = None
    fecha_debug = None

    # Token-level 'FECHA' label detection (high priority): handles OCR that splits
    # the label and the date into separate tokens (e.g. ['FECHA', '12', '/', '05', '/', '2023']).
    if tokens:
        try:
            for idx, tk in enumerate(tokens):
                t0 = (tk.get('text') or '').strip()
                if 'fecha' in t0.lower():
                    # collect following tokens that look like digits or slashes/punct
                    seq = []
                    for j in range(idx + 1, min(len(tokens), idx + 9)):
                        tj = (tokens[j].get('text') or '').strip()
                        if tj == '':
                            continue
                        seq.append(tj)
                        joined = ''.join(seq)
                        # look for dd/mm/yyyy or yyyy/mm/dd inside the joined tokens
                        m = re.search(r"(\d{1,4}/\d{1,2}/\d{1,4})", joined)
                        if m:
                            cand = m.group(1)
                            parts = cand.split('/')
                            if any(len(p) == 4 for p in parts):
                                # coerce into ISO
                                try:
                                    if len(parts[0]) == 4:
                                        yyyy = int(parts[0]); mm = int(parts[1]); dd = int(parts[2])
                                    else:
                                        dd = int(parts[0]); mm = int(parts[1]); yyyy = int(parts[2]) if len(parts[2]) == 4 else _normalize_year(int(parts[2]))
                                    if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2099:
                                        fecha = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
                                        fecha_debug = [fecha]
                                        break
                                except Exception:
                                    pass
                    if fecha:
                        break
        except Exception:
            # be defensive; fall back to text-level detection below
            fecha = None
            fecha_debug = None

    # Text-level detection (includes prioritized 'fecha' label in text)
    if not fecha:
        dates = _find_dates(texto_clean)
        if dates:
            fecha = dates[0]
            fecha_debug = dates

    # token-level fallback for dates (if still not found): examine individual tokens
    if not fecha and tokens:
        token_dates = []
        for tk in tokens:
            t = (tk.get('text') or '').strip()
            conf = tk.get('conf') or 0
            # direct slash-containing token (e.g. '12/05/2023') -> prefer if year 4-digit
            if '/' in t:
                m = re.search(r"(\d{1,4}/\d{1,2}/\d{1,4})", t)
                if m:
                    cand = m.group(1)
                    parts = cand.split('/')
                    if any(len(p) == 4 for p in parts):
                        v = None
                        try:
                            if len(parts[0]) == 4:
                                yyyy = int(parts[0]); mm = int(parts[1]); dd = int(parts[2])
                            else:
                                dd = int(parts[0]); mm = int(parts[1]); yyyy = int(parts[2]) if len(parts[2]) == 4 else _normalize_year(int(parts[2]))
                            if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2099:
                                v = f"{yyyy:04d}-{mm:02d}-{dd:02d}"
                        except Exception:
                            v = None
                        if v:
                            token_dates.append((conf, v))
            # otherwise try generic regex on token
            for pat in _DATE_RE_LIST:
                m = pat.search(t)
                if m:
                    v = _parse_date_match(m)
                    if v:
                        token_dates.append((conf, v))
        if token_dates:
            token_dates.sort(key=lambda x: x[0], reverse=True)
            fecha = token_dates[0][1]
            fecha_debug = [td[1] for td in token_dates]

    # Monto
    monto_val, monto_raw, monto_debug = _find_amounts(texto_clean, tokens)

    # Merchant: conservative choice among top lines
    def looks_fiscal(s: str) -> bool:
        s2 = s.upper()
        fiscal_terms = ['CUIT', 'C.U.I.T', 'CIF', 'FACTURA', 'NRO', 'Nº', 'IVA']
        return any(t in s2 for t in fiscal_terms) or bool(re.search(r"\b\d{4,}\b", s2))

    merchant = None
    for ln in texto_lines[:6]:
        letters = sum(1 for c in ln if c.isalpha())
        digits = sum(1 for c in ln if c.isdigit())
        if letters > digits and letters >= 3 and not looks_fiscal(ln):
            merchant = ln.strip()
            break

    # token-based fallback
    if not merchant and tokens:
        best_token = None
        best_conf = -1
        for tk in tokens:
            t = (tk.get('text') or '').strip()
            if sum(1 for c in t if c.isalpha()) >= 3 and not any(c.isdigit() for c in t):
                if (tk.get('conf') or 0) > best_conf:
                    best_conf = tk.get('conf') or 0
                    best_token = t
        if best_token:
            merchant = re.sub(r"\b(S\.A\.|SA\.|TEL:)?.*$", "", best_token, flags=re.IGNORECASE).strip(' ,.-:;')

    # Detected items / category hint
    detected_items: List[str] = []
    try:
        up = (texto_clean or '').upper()
        for kw in _PRODUCT_KEYWORDS:
            if kw in up:
                detected_items.append(kw)
        # fuzzy stems
        if not detected_items:
            stems = ['CAPU', 'MEDIAL', 'CAF', 'CROISS', 'EMPAN']
            for s in stems:
                if s in up:
                    detected_items.append(s)
    except Exception:
        detected_items = []

    category_hint = 'comida' if detected_items else None

    result: Dict[str, Any] = {"fecha": fecha, "monto": monto_val, "texto": original_text, "texto_clean": texto_clean, "texto_lines": texto_lines}
    if merchant:
        result["merchant"] = merchant
    if monto_raw:
        result["monto_raw"] = monto_raw
    if monto_debug:
        result["monto_debug"] = monto_debug
    if fecha_debug:
        result["fecha_debug"] = fecha_debug
    if detected_items:
        result['detected_items'] = detected_items
    if category_hint:
        result['category_hint'] = category_hint
    return result
