"""OCR analysis helpers: extract date, amount and merchant from OCR output.

Small, robust implementation used by the FastAPI OCR endpoint. Returns
debuggable fields when debug_tokens is enabled upstream.
"""

import re
from typing import Optional, Dict, Any, List
import math
from collections import defaultdict


# Simple regexes
_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})\b")
"""OCR analysis helpers: extract date, amount and merchant from OCR output.

Small, robust implementation used by the FastAPI OCR endpoint. Returns
debuggable fields when debug_tokens is enabled upstream.
"""

import re
from typing import Optional, Dict, Any, List
import math
from collections import defaultdict


# Simple regexes
_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})\b")
_SIMPLE_AMOUNT_RE = re.compile(r"\b\d+[\.,]?\d*\b")
_KEYWORDS = ["TOTAL", "TOT", "IMPORTE", "PAGO", "MONTO"]
_DATE_RE_LIST = [
    re.compile(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b"),
    re.compile(r"\b(\d{2})[/-](\d{2})[/-](\d{2})\b"),
    re.compile(r"\b(\d{4})[/-](\d{2})[/-](\d{2})\b"),
]


def _normalize_year(n: int) -> int:
    return 2000 + n if n < 50 else 1900 + n


def _parse_date_match(m: re.Match) -> Optional[str]:
    try:
        a, b, c = m.groups()
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
    for pat in _DATE_RE_LIST:
        for m in pat.finditer(text):
            v = _parse_date_match(m)
            if v:
                out.append(v)
    return out


def _find_amounts(text: str) -> List[str]:
    found: List[str] = [m.group() for m in _AMOUNT_RE.finditer(text)]
    if not found:
        found = [m.group() for m in _SIMPLE_AMOUNT_RE.finditer(text)]
    toks = [t for t in re.split(r"\s+", text) if t]
    for i in range(len(toks)):
        for j in range(i, min(i + 3, len(toks))):
            cand = "".join(toks[i:j+1])
            if re.search(r"\d", cand) and re.search(r"[\.,]", cand):
                if _AMOUNT_RE.search(cand):
                    found.append(cand)
            else:
                if _SIMPLE_AMOUNT_RE.fullmatch(cand):
                    found.append(cand)
    seen = set(); dedup: List[str] = []
    for a in found:
        if a not in seen:
            seen.add(a); dedup.append(a)
    return dedup


def extraer_campos(texto: str, tokens: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract fecha (ISO), monto (float) and merchant when possible.

    Returns debug fields monto_raw, monto_debug, fecha_debug when available.
    """
    if not texto:
        return {"fecha": None, "monto": None, "texto": ""}

    original_text = texto
    texto_lines = [ln.rstrip() for ln in original_text.splitlines() if ln.strip()]
    texto_clean = "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in texto_lines)

    fecha = None; fecha_debug = None
    dates = _find_dates(texto_clean)
    if dates:
        fecha = dates[0]; fecha_debug = dates
    # token-level date fallback (useful when OCR text aggregation mangles date)
    if not fecha and tokens:
        token_dates = []
        for tk in tokens:
            t = (tk.get("text") or "").strip()
            conf = tk.get("conf") or 0
            for pat in _DATE_RE_LIST:
                m = pat.search(t)
                if m:
                    v = _parse_date_match(m)
                    if v:
                        token_dates.append((conf, v, t))
        if token_dates:
            token_dates.sort(key=lambda x: x[0], reverse=True)
            fecha = token_dates[0][1]
            fecha_debug = [td[1] for td in token_dates]

    amounts = _find_amounts(texto_clean)
    monto_val: Optional[float] = None
    monto_raw: Optional[str] = None
    monto_debug: List[Dict[str, Any]] = []

    def _norm_num(s: str) -> str:
        only = re.sub(r"[^0-9,\.]", "", s)
        if only.count(",") == 1 and only.count(".") >= 1:
            return only.replace(".", "").replace(",", ".")
        return only.replace(",", ".")

    # Prefer token-level candidates — improved: combine nearby tokens and prefer proximity to TOTAL
    if tokens:
        # helper: if an integer candidate has only 2 digits, try to find a single-digit
        # token immediately to its left on the same baseline and prepend it (e.g. '35' + '5' -> '355')
        def _try_prepend_digit(int_text: str, base_token: dict):
            """Try to extend a 2-digit integer by prepending or appending a nearby single digit.

            Returns the extended string if a good candidate is found, otherwise returns the
            original int_text. Uses `geom_tokens` for spatial candidates and scores by
            confidence and proximity. This is conservative: only attempts when int_text
            contains exactly 2 digits.
            """
            try:
                if not int_text or len(re.sub(r"[^0-9]", "", int_text)) != 2:
                    return int_text
                base_left = (base_token.get('left', 0) or 0)
                base_right = base_left + (base_token.get('w', base_token.get('width', 0)) or 0)
                base_top = (base_token.get('top', 0) or 0)

                best_prep = None; best_prep_score = -1
                best_app = None; best_app_score = -1

                for g in geom_tokens:
                    gtxt = re.sub(r"[^0-9]", "", g.get('text') or "")
                    if not re.fullmatch(r"\d", gtxt):
                        continue
                    g_left = (g.get('left', 0) or 0)
                    g_right = g_left + (g.get('w', g.get('width', 0)) or 0)
                    g_top = (g.get('top', 0) or 0)
                    # vertical proximity check (allow generous slack for noisy receipts)
                    vdist = abs(g_top - base_top)
                    if vdist > 100:
                        continue
                    conf = (g.get('conf') or 0)
                    # consider candidate to PREPEND (digit left of base)
                    if g_right <= base_left + 40:
                        horiz = base_left - g_right
                        score = conf - (horiz * 0.02) - (vdist * 0.01)
                        if score > best_prep_score:
                            best_prep_score = score; best_prep = gtxt
                    # consider candidate to APPEND (digit right of base)
                    if g_left >= base_right - 40:
                        horiz = g_left - base_right
                        score = conf - (horiz * 0.02) - (vdist * 0.01)
                        if score > best_app_score:
                            best_app_score = score; best_app = gtxt

                # prefer prepend when scores similar (visual alignment tends to put leading digit left)
                if best_prep and (best_prep_score >= best_app_score - 5):
                    return best_prep + int_text
                if best_app:
                    return int_text + best_app
            except Exception:
                pass
            return int_text
        token_amounts = []
        # collect numeric-ish tokens with their geometry
        geom_tokens = []
        for idx, tk in enumerate(tokens):
            txt = (tk.get("text") or "").strip()
            conf = tk.get("conf") or 0
            if not txt or not any(ch.isdigit() for ch in txt):
                continue
            left = tk.get("left", 0) or 0
            top = tk.get("top", 0) or 0
            width = tk.get("width", 0) or 0
            geom_tokens.append({"idx": idx, "text": txt, "conf": conf, "left": left, "top": top, "w": width})

        # find TOTAL tokens for proximity scoring
        total_tokens = [tk for tk in tokens if any(k in (tk.get("text") or "").upper() for k in _KEYWORDS)]

        # group tokens by approximate line (top within 8 px)
        lines = []
        for t in sorted(geom_tokens, key=lambda x: (x["top"], x["left"])):
            if not lines or abs(t["top"] - lines[-1][0]["top"]) > 8:
                lines.append([t])
            else:
                lines[-1].append(t)

        candidates = []
        # create candidate strings by joining up to 3 adjacent tokens in a line
        for line in lines:
            line_sorted = sorted(line, key=lambda x: x["left"])
            n = len(line_sorted)
            for i in range(n):
                for j in range(i, min(n, i + 3)):
                    seq = line_sorted[i : j + 1]
                    # require tokens to be horizontally adjacent (small gap)
                    contiguous = True
                    for k in range(len(seq) - 1):
                        right_edge = seq[k]["left"] + seq[k]["w"]
                        gap = seq[k + 1]["left"] - right_edge
                        if gap > 40:  # tokens too far apart, likely different columns (relaxed)
                            contiguous = False
                            break
                    if not contiguous:
                        continue
                    s = "".join(t["text"] for t in seq)
                    if not any(ch.isdigit() for ch in s):
                        continue
                    # accept if looks like amount
                    if _AMOUNT_RE.search(s) or re.search(r"[\.,]\d{2}$", s) or _SIMPLE_AMOUNT_RE.fullmatch(s):
                        # reject joining multiple numeric tokens that look like IDs: if sequence length>1, none contain decimal sep and all tokens have >=3 digits
                        if len(seq) > 1:
                            any_decimal = any(re.search(r"[\.,]", t["text"]) for t in seq)
                            token_digit_lens = [len(re.sub(r"[^0-9]", "", t["text"])) for t in seq]
                            if (not any_decimal) and all(d >= 3 for d in token_digit_lens):
                                # likely an ID; skip
                                continue
                        avg_conf = sum(t["conf"] for t in seq) / len(seq)
                        # compute min distance to any TOTAL token center
                        min_dist = 1e9
                        for tt in total_tokens:
                            tx = (tt.get("left", 0) or 0) + (tt.get("width", 0) or 0) / 2
                            ty = (tt.get("top", 0) or 0) + (tt.get("height", 0) or 0) / 2
                            cx = sum(t["left"] + t["w"]/2 for t in seq) / len(seq)
                            cy = sum(t["top"] for t in seq) / len(seq)
                            min_dist = min(min_dist, math.hypot(cx - tx, cy - ty))
                        candidates.append({"text": s, "conf": avg_conf, "digits": sum(1 for c in s if c.isdigit()), "dist": min_dist})

            # Special-case: sometimes OCR produces a line like "TOTAL . .37" (decimal fragment)
            # and the integer part is a separate token (e.g. "35") to the right. Detect this
            # pattern and combine the nearby integer token with the decimal fragment to a
            # proper amount before scoring candidates.
            if not monto_val and total_tokens:
                # look for decimal-only fragments in the cleaned text (e.g. ".37" or ",37" possibly spaced or with extra dots)
                dec_frag_match = re.search(r"[\.,]\s*\.?\s*(\d{2})\b", texto_clean)
                if dec_frag_match:
                    dec = dec_frag_match.group(1)
                    # choose the most likely TOTAL token as reference
                    tt = sorted(total_tokens, key=lambda x: (x.get("conf") or 0), reverse=True)[0]
                    t_left = (tt.get("left", 0) or 0)
                    t_top = (tt.get("top", 0) or 0)

                    # collect candidate numeric tokens in a broad region right of TOTAL
                    candidates_right = [g for g in geom_tokens if abs(g.get('top', 0) - t_top) <= 80 and g.get('left', 0) > t_left - 20]
                    # sort by left coordinate (visual order)
                    candidates_right = sorted(candidates_right, key=lambda x: x.get('left', 0))

                    # build concatenations of up to 3 adjacent numeric tokens (allowing gaps)
                    assembled_candidates = []
                    for i in range(len(candidates_right)):
                        cur = re.sub(r"[^0-9]", "", candidates_right[i].get('text') or "")
                        cur_conf = (candidates_right[i].get('conf') or 0)
                        cur_left = candidates_right[i].get('left', 0) or 0
                        cur_top = candidates_right[i].get('top', 0) or 0
                        if not cur:
                            continue
                        assembled_candidates.append({'text': cur, 'conf': cur_conf, 'left': cur_left, 'top': cur_top})
                        # try to append the next one or two tokens if they are close enough
                        for j in range(i+1, min(i+3, len(candidates_right))):
                            nxt = candidates_right[j]
                            gap = nxt.get('left', 0) - (candidates_right[j-1].get('left', 0) + (candidates_right[j-1].get('w', candidates_right[j-1].get('width',0)) or 0))
                            vslack = abs(nxt.get('top',0) - cur_top)
                            if gap is None:
                                gap = 0
                            # allow moderately large gaps created by OCR fragmentation
                            if gap <= 140 and vslack <= 60:
                                cur += re.sub(r"[^0-9]", "", nxt.get('text') or "")
                                cur_conf = max(cur_conf, (nxt.get('conf') or 0))
                                assembled_candidates.append({'text': cur, 'conf': cur_conf, 'left': cur_left, 'top': cur_top})
                            else:
                                break

                    # also consider single-digit tokens to the left of assembled ones (to prepend)
                    best_candidate = None; best_score = -1
                    for ac in assembled_candidates:
                        # try raw assembled and also attempt to prepend/appending neighbouring single digits
                        cand_variants = [ac['text']]
                        # look for single-digit tokens left of the candidate within 140px
                        for g in geom_tokens:
                            gtxt = re.sub(r"[^0-9]", "", g.get('text') or "")
                            if not re.fullmatch(r"\d", gtxt):
                                continue
                            if g.get('left',0) + (g.get('w', g.get('width', 0)) or 0) <= ac['left'] + 140 and abs(g.get('top',0) - ac['top']) <= 100:
                                cand_variants.append(gtxt + ac['text'])
                            if g.get('left',0) >= ac['left'] - 40 and g.get('left',0) <= ac['left'] + 200 and abs(g.get('top',0) - ac['top']) <= 100:
                                cand_variants.append(ac['text'] + gtxt)

                        for var in set(cand_variants):
                            # skip unrealistic longs
                            if len(var) > 4:
                                continue
                            # avoid zero-padded ids unless very near TOTAL
                            if re.match(r"^0\d+$", var):
                                if abs(ac['top'] - t_top) > 30:
                                    continue
                            # score: confidence + digits bonus - horizontal distance penalty
                            horiz_dist = abs((ac.get('left',0) or 0) - t_left)
                            score = (ac.get('conf') or 0) * 1.5 + len(var) * 6 - horiz_dist * 0.02
                            # small boost if variant ends with the same last digit as many nearby tokens (stability)
                            if re.search(r"\d$", var):
                                last = var[-1]
                                similar_last = sum(1 for g in geom_tokens if (re.sub(r"[^0-9]", "", g.get('text') or "") or '').endswith(last))
                                score += min(similar_last, 3) * 2
                            if score > best_score:
                                best_score = score; best_candidate = var

                    if best_candidate:
                        # try to normalize and set monto
                        combined = f"{best_candidate},{dec}"
                        try:
                            monto_val = float(_norm_num(combined))
                        except Exception:
                            monto_val = None
                        if monto_val is not None:
                            monto_raw = combined
                            monto_debug.append({"chosen": monto_raw, "parsed": monto_val, "method": "total_fragment_multi_assembled"})

        # Additional heuristic: look for decimal fragment tokens (e.g. ".37") or
        # tokens ending with a decimal part near the TOTAL token, and try to
        # combine them with nearby integer tokens (even if on different lines).
        # This uses a larger search radius around TOTAL to reconstruct dispersed totals.
        if not monto_val and total_tokens:
            for tt in total_tokens:
                tx_center = (tt.get("left", 0) or 0) + (tt.get("width", 0) or 0) / 2
                ty_center = (tt.get("top", 0) or 0) + (tt.get("height", 0) or 0) / 2
                # search tokens in a broad box around TOTAL
                nearby = [g for g in geom_tokens if abs((g.get("top", 0) or 0) - ty_center) <= 250 or abs((g.get("left", 0) or 0) - tx_center) <= 500]
                # decimal-like tokens ('.37', ',37', or tokens that end with ',dd')
                dec_tokens = [g for g in nearby if re.match(r'^[\.,]\d{1,2}$', g.get("text", "")) or re.search(r'[\.,]\d{2}$', g.get("text", ""))]
                # integer-like tokens up to 4 digits (avoid long IDs)
                # prefer integer-like tokens up to 3 digits for total integers (avoid 4-digit IDs)
                int_tokens = [g for g in nearby if 1 <= len(re.sub(r"[^0-9]", "", g.get("text", ""))) <= 3]
                # try to pair each decimal fragment with the closest integer token horizontally
                for dec in dec_tokens:
                    best_int = None; best_dist = 1e9
                    dec_cx = dec.get("left", 0) + (dec.get("w", dec.get("width", 0)) or 0) / 2 if dec.get("left") is not None else dec.get("left", 0)
                    for it in int_tokens:
                        it_cx = it.get("left", 0) + (it.get("w", it.get("width", 0)) or 0) / 2 if it.get("left") is not None else it.get("left", 0)
                        d = abs(it_cx - dec_cx)
                        if d < best_dist:
                            best_dist = d; best_int = it
                    if best_int:
                        intpart = re.sub(r"[^0-9]", "", best_int.get("text", ""))
                        # try to extend 2-digit integer by prepending a nearby single digit
                        if intpart:
                            intpart = _try_prepend_digit(intpart, best_int)
                        # avoid picking integers that look like zero-padded IDs (e.g. '0002')
                        if re.match(r"^0\d+$", intpart):
                            # only accept zero-padded integer if it's very close vertically to TOTAL
                            allow_zero_padded = False
                            if total_tokens:
                                for tt in total_tokens:
                                    if abs(best_int.get('top', 0) - (tt.get('top', 0) or 0)) <= 30:
                                        allow_zero_padded = True; break
                            if not allow_zero_padded:
                                continue
                        dec_m = re.search(r"(\d{1,2})$", dec.get("text", ""))
                        if intpart and dec_m:
                            combined = f"{intpart},{dec_m.group(1)}"
                            try:
                                monto_val = float(_norm_num(combined))
                            except Exception:
                                monto_val = None
                            if monto_val is not None:
                                monto_raw = combined
                                monto_debug.append({"chosen": monto_raw, "parsed": monto_val, "method": "total_nearby_combination"})
                                break
                if monto_val:
                    break

        # Aggressive fallback: pair any decimal-fragment token (e.g. ".37") with
        # the nearest integer-like token anywhere in the page (within a distance
        # threshold). This is risky (more false positives) but can recover totals
        # when layout is badly mangled.
        if not monto_val:
            # collect decimal fragments and integer candidates
            dec_frags = [g for g in geom_tokens if re.match(r'^[\.,]\d{1,2}$', g['text']) or re.search(r'[\.,]\d{2}$', g['text'])]
            # be stricter: prefer integer candidates of 1-3 digits to avoid selecting multi-digit IDs like '3851'
            int_cands = [g for g in geom_tokens if 1 <= len(re.sub(r'[^0-9]', '', g['text'])) <= 3]
            if dec_frags and int_cands:
                for dec in dec_frags:
                    # if we have TOTAL tokens, require the decimal fragment to be reasonably near TOTAL
                    if total_tokens:
                        dec_top = dec.get('top', 0) or 0
                        min_vert_to_total = min(abs((tt.get('top', 0) or 0) - dec_top) for tt in total_tokens)
                        # skip decimal fragments that are far vertically from any TOTAL (likely unrelated)
                        if min_vert_to_total > 300:
                            continue

                    # compute center for dec
                    dec_cx = dec['left'] + dec['w']/2
                    dec_cy = dec['top']
                    best = None; best_d = 1e9
                    for it in int_cands:
                        it_cx = it['left'] + it['w']/2
                        it_cy = it['top']
                        d = math.hypot(it_cx - dec_cx, it_cy - dec_cy)
                        if d < best_d:
                            best_d = d; best = it
                    # only accept if reasonably close (tunable threshold)
                    # reduce the global radius for aggressive pairing to avoid far-away false matches
                    # if best is acceptable and within reduced distance threshold
                    if best and best_d < 600:
                        intpart = re.sub(r'[^0-9]', '', best['text'])
                        # try to extend 2-digit integer by prepending a nearby single digit
                        if intpart:
                            intpart = _try_prepend_digit(intpart, best)
                        # avoid pairing with zero-padded numeric tokens (likely IDs) unless they're very near TOTAL
                        if re.match(r'^0\d+$', intpart):
                            allow_zero = False
                            if total_tokens:
                                dec_top_local = dec.get('top', 0) or 0
                                min_vert = min(abs((tt.get('top', 0) or 0) - dec_top_local) for tt in total_tokens)
                                if min_vert <= 30:
                                    allow_zero = True
                            if not allow_zero:
                                continue
                        dec_m = re.search(r'(\d{1,2})$', dec['text'])
                        if intpart and dec_m:
                            # require integer token to be left or not far to the right of the decimal fragment
                            dec_cx_local = dec['left'] + dec['w']/2
                            int_cx_local = best['left'] + best['w']/2
                            if int_cx_local > dec_cx_local + 200:
                                # integer is too far right, skip to avoid wrong pairing
                                continue
                            combined = f"{intpart},{dec_m.group(1)}"
                            try:
                                monto_val = float(_norm_num(combined))
                            except Exception:
                                monto_val = None
                            if monto_val is not None:
                                monto_raw = combined
                                monto_debug.append({"chosen": monto_raw, "parsed": monto_val, "method": "aggressive_combination", "dist": best_d})
                                break

        # scoring favoring decimal tokens, confidence, digits and proximity to TOTAL
        def cand_score(c):
            s = (c["conf"] or 0) * 2 + (c["digits"] or 0) * 4
            if re.search(r"[\.,]\d{2}$", c["text"]):
                s += 60
            # proximity bonus
            if c["dist"] < 1e9:
                s += max(0, 120 - c["dist"]) / 2
            # reject long digit-only candidates without decimal separators
            digits_only = re.sub(r"[^0-9]", "", c["text"])
            if len(digits_only) > 5 and not re.search(r"[\.,]", c["text"]):
                s -= 1000
            # penalize long ids additionally
            if len(digits_only) >= 8 and not re.search(r"[\.,]\d{2}$", c["text"]):
                s -= 200
            return s

        if candidates:
            # If the aggregated OCR text contains explicit decimal-style amounts (e.g. 355,37)
            # prefer those by promoting them over token-only numeric sequences which often
            # represent IDs found at the bottom of receipts. We still allow a token candidate
            # with an explicit decimal to win if it scores higher.
            text_amounts = [m.group() for m in _AMOUNT_RE.finditer(texto_clean)]
            # mark candidates that have a decimal separator
            for c in candidates:
                c["has_dec"] = bool(re.search(r"[\.,]\d{2}$", c["text"]))

            # if text has good decimal amounts, compute a boosted score for those
            if text_amounts:
                # normalize text amounts for quick lookup (remove thousands separators)
                norm_text_amounts = set(_norm_num(a) for a in text_amounts)
                # boost candidates that match a text-level decimal amount exactly
                for c in candidates:
                    try:
                        if _norm_num(c["text"]) in norm_text_amounts:
                            c["score_boost"] = 200
                        else:
                            c["score_boost"] = 0
                    except Exception:
                        c["score_boost"] = 0

            candidates.sort(key=lambda x: cand_score(x) + (x.get("score_boost", 0) or 0), reverse=True)
            best = candidates[0]
            monto_raw = best["text"]
            try:
                monto_val = float(_norm_num(monto_raw))
            except Exception:
                monto_val = None
            monto_debug.append({"chosen": monto_raw, "parsed": monto_val, "method": "token"})

    # fallback to text-level aggregation
    if monto_val is None and amounts:
        agg = defaultdict(lambda: {"count": 0, "lines": []})
        for i, ln in enumerate(texto_lines):
            for m in _AMOUNT_RE.finditer(ln):
                s = m.group(); agg[s]["count"] += 1; agg[s]["lines"].append(i)
        best_score = -1; best_candidate = None
        for k, info in agg.items():
            has_dec = 1 if re.search(r"[\.,]\d{2}$", k) else 0
            occ = info["count"]; min_dist = min(info["lines"]) if info["lines"] else 9999
            score = (has_dec * 200) + (occ * 50) + max(0, 100 - min_dist * 10)
            if score > best_score: best_score = score; best_candidate = k
        if best_candidate:
            monto_raw = best_candidate
            try: monto_val = float(_norm_num(best_candidate))
            except Exception: monto_val = None
            monto_debug.append({"chosen": monto_raw, "parsed": monto_val, "method": "text_agg"})

    # Forced reconstruction (conservative): if we still don't have a monto_val
    # but the page contains the word TOTAL and a high-confidence decimal fragment
    # (e.g. ".37" with conf >= 50) plus an integer-like token to its left within
    # a reasonable vertical band, construct the combined amount and prefer it.
    if monto_val is None and tokens:
        try:
            total_tokens = [tk for tk in tokens if any(k in (tk.get("text") or "").upper() for k in _KEYWORDS)]
            if total_tokens:
                # collect decimal fragments from tokens with decent confidence
                dec_tokens = [tk for tk in tokens if re.fullmatch(r"[\.,]\d{2}", (tk.get('text') or '').strip()) and (tk.get('conf') or 0) >= 50]
                # if none found, try weaker ones from texto_clean
                if not dec_tokens:
                    m = re.search(r"([\.,]\s*\.?\s*(\d{2}))\b", texto_clean)
                    if m:
                        # create a pseudo-token with moderate confidence placed near the TOTAL token
                        # so spatial pairing prefers integers close to TOTAL instead of distant IDs
                        tt_ref = sorted(total_tokens, key=lambda x: (x.get('conf') or 0), reverse=True)[0]
                        tt_left = (tt_ref.get('left', 0) or 0) + (tt_ref.get('width', 0) or 0)
                        tt_top = (tt_ref.get('top', 0) or 0)
                        dec_tokens = [{"text": m.group(1), "conf": 50, "left": tt_left, "top": tt_top, "width": 0, "height": 0}]

                if dec_tokens:
                    # search for integer-like tokens to the left within vertical tolerance
                    for dec in dec_tokens:
                        dec_left = dec.get('left', 0) or 0
                        dec_top = dec.get('top', 0) or 0
                        # integer candidates: 1-3 digits and reasonable confidence (>10)
                        int_cands = [tk for tk in tokens if re.fullmatch(r"\d{1,3}", (tk.get('text') or '').strip()) and (tk.get('conf') or 0) >= 10]
                        best = None; best_score = None
                        for it in int_cands:
                            it_left = it.get('left', 0) or 0
                            it_top = it.get('top', 0) or 0
                            # compute vertical separation first
                            vert = abs(it_top - dec_top)
                            # require integer to be left of (or overlapping) dec or at least not far to the right
                            if it_left > dec_left + 200:
                                continue
                            # allow larger vertical separation for heavily mangled receipts
                            # (was 500px; relax to 1200px to allow matching fragments like '35' + '.37')
                            if vert > 1200:
                                # still avoid pairing tokens that are extremely far apart
                                continue
                            horiz = max(0, dec_left - (it_left + (it.get('width') or 0)))
                            # score favors proximity and confidence and digit length
                            score = (it.get('conf') or 0) * 1.5 + max(0, 50 - vert) - horiz
                            digits = len(re.sub(r"[^0-9]", "", it.get('text') or ""))
                            score += digits * 5
                            if best_score is None or score > best_score:
                                best_score = score; best = it
                        if best:
                            intpart = re.sub(r"[^0-9]", "", best.get('text') or "")
                            # extract 1-2 digit decimal part more robustly
                            # try to extend 2-digit integers with a nearby single digit
                            if intpart:
                                intpart = _try_prepend_digit(intpart, best)
                            dec_m = re.search(r"(\d{1,2})", dec.get('text') or "")
                            dec_digits = dec_m.group(1) if dec_m else None
                            if intpart and dec_digits:
                                combined = f"{intpart},{dec_digits}"
                                try:
                                    val = float(_norm_num(combined))
                                except Exception:
                                    val = None
                                if val is not None:
                                    monto_val = val
                                    monto_raw = combined
                                    monto_debug.append({"chosen": monto_raw, "parsed": monto_val, "method": "forced_total_reconstruction_relaxed"})
                                    break
        except Exception:
            pass

    # Override heuristic: if we already picked a monto but it's spatially far from
    # the TOTAL keyword while there exists a decimal fragment + integer near
    # TOTAL, prefer the reconstructed combined amount. This helps when Tesseract
    # finds a clearly formatted amount elsewhere (e.g. '208,00') but the true
    # printed total is split (e.g. '35' + '.37') around the TOTAL label.
    try:
        if tokens:
            total_tokens = [tk for tk in tokens if any(k in (tk.get("text") or "").upper() for k in _KEYWORDS)]
            if total_tokens:
                # locate current chosen token (if any) to compute distance to TOTAL
                current_token = None
                if monto_raw:
                    norm_target = re.sub(r"[^0-9]", "", monto_raw)
                    for tk in tokens:
                        if re.sub(r"[^0-9]", "", (tk.get("text") or "")) == norm_target:
                            current_token = tk; break

                # find a decimal fragment token (strong preference to tokenized one)
                dec_token = None
                dec_candidates = [tk for tk in tokens if re.fullmatch(r"[\.,]\d{1,2}", (tk.get('text') or '').strip()) and (tk.get('conf') or 0) >= 40]
                if not dec_candidates:
                    m = re.search(r"([\.,]\s*\.?\s*(\d{2}))\b", texto_clean)
                    if m:
                        # place pseudo-decimal near the most confident TOTAL token so pairing is local
                        tt_ref = sorted(total_tokens, key=lambda x: (x.get('conf') or 0), reverse=True)[0]
                        tt_left = (tt_ref.get('left', 0) or 0) + (tt_ref.get('width', 0) or 0)
                        tt_top = (tt_ref.get('top', 0) or 0)
                        dec_candidates = [{"text": m.group(1), "conf": 45, "left": tt_left, "top": tt_top, "width": 0, "height": 0}]
                if dec_candidates:
                    # choose highest-conf decimal fragment
                    dec_token = sorted(dec_candidates, key=lambda x: (x.get('conf') or 0), reverse=True)[0]

                # find integer candidate near TOTAL (allow generous vertical/horizontal tolerances)
                if dec_token:
                    # use the top-most TOTAL token as reference
                    tt = sorted(total_tokens, key=lambda x: (x.get('conf') or 0), reverse=True)[0]
                    t_left = (tt.get('left', 0) or 0); t_top = (tt.get('top', 0) or 0)
                    # accept integer-like tokens of 1-3 digits (prefer shorter integers for totals)
                    int_cands = [tk for tk in tokens if re.fullmatch(r"\d{1,3}", (tk.get('text') or '').strip())]
                    best_int = None; best_score = -1
                    for it in int_cands:
                        it_left = it.get('left', 0) or 0; it_top = it.get('top', 0) or 0
                        # avoid zero-padded numeric tokens (likely IDs) unless they sit very near TOTAL
                        int_text_digits = re.sub(r"[^0-9]", "", it.get('text') or "")
                        if re.match(r"^0\d+$", int_text_digits):
                            near_total = False
                            for tt2 in total_tokens:
                                if abs(it_top - (tt2.get('top', 0) or 0)) <= 30:
                                    near_total = True; break
                            if not near_total:
                                continue
                        vert = abs(it_top - (dec_token.get('top', 0) or 0))
                        horiz = abs((it_left + (it.get('width') or 0)/2) - ((dec_token.get('left', 0) or 0) + ((dec_token.get('width') or 0)/2)))
                        # scoring: prefer closer and moderately confident integers
                        sc = (it.get('conf') or 0) * 1.2 - vert * 0.01 - horiz * 0.02 + len(int_text_digits) * 4
                        if sc > best_score:
                            best_score = sc; best_int = it

                    if best_int:
                        intpart = re.sub(r"[^0-9]", "", best_int.get('text') or "")
                        dec_m = re.search(r"(\d{1,2})", dec_token.get('text') or "")
                        dec_digits = dec_m.group(1) if dec_m else None
                        if intpart and dec_digits:
                            combined = f"{intpart},{dec_digits}"
                            try:
                                combined_val = float(_norm_num(combined))
                            except Exception:
                                combined_val = None
                            if combined_val is not None:
                                # compute distance of current token to TOTAL (if available)
                                far_enough = True
                                if current_token:
                                    ct_cx = (current_token.get('left', 0) or 0) + (current_token.get('width', 0) or 0)/2
                                    ct_cy = (current_token.get('top', 0) or 0) + (current_token.get('height', 0) or 0)/2
                                    tt_cx = t_left + (tt.get('width', 0) or 0)/2
                                    tt_cy = t_top + (tt.get('height', 0) or 0)/2
                                    dist_curr = math.hypot(ct_cx - tt_cx, ct_cy - tt_cy)
                                    # if current chosen amount is far (>400px) from TOTAL, prefer reconstructed
                                    if dist_curr <= 400:
                                        far_enough = False
                                if far_enough:
                                    monto_val = combined_val
                                    monto_raw = combined
                                    monto_debug.append({"chosen": monto_raw, "parsed": monto_val, "method": "forced_total_reconstruction_override"})
    except Exception:
        # be conservative: don't crash analysis on unexpected token shapes
        pass

    # merchant extraction: prefer token clusters near top, else first non-fiscal top line
    merchant = None
    def looks_fiscal(s: str) -> bool:
        stop = ["RESPONSABLE", "IVA", "CONSUMIDOR", "FINAL", "RUC", "INSCRIPTO", "CUIT", "CUIL"]
        return any(w in (s or "").upper() for w in stop)

    if tokens:
        # Extra heuristic: if we ended up with a two-digit integer + decimal (e.g. "53,37")
        # try a broader prepend search for any high-confidence single-digit token on the
        # page (relaxed vertical/horizontal tolerances). This helps when the leading
        # digit is fragmented or OCR'd separately from the 2-digit part.
        try:
            if monto_raw and re.fullmatch(r"\d{2}[\.,]\d{2}", monto_raw):
                # locate decimal digits and current integer part
                parts = re.split(r"[\.,]", monto_raw)
                if parts and len(parts) >= 2:
                    cur_int = re.sub(r"[^0-9]", "", parts[0])
                    cur_dec = re.sub(r"[^0-9]", "", parts[1])
                    if len(cur_int) == 2:
                        # search for the best single-digit to PREPEND or APPEND.
                        best_variant = None
                        best_variant_score = -1
                        # reference position: try to find the token we used for the current amount
                        ref_token = None
                        norm_target = cur_int
                        for tk in tokens:
                            if re.sub(r"[^0-9]", "", (tk.get('text') or "")) == norm_target:
                                ref_token = tk; break
                        # choose a reasonable reference point (either the integer token or the most confident TOTAL)
                        ref_left = None; ref_top = None
                        if ref_token:
                            ref_left = (ref_token.get('left', 0) or 0) + ((ref_token.get('width', 0) or 0) / 2)
                            ref_top = ref_token.get('top', 0) or 0
                        else:
                            if total_tokens:
                                tt = sorted(total_tokens, key=lambda x: (x.get('conf') or 0), reverse=True)[0]
                                ref_left = (tt.get('left', 0) or 0) + ((tt.get('width', 0) or 0) / 2)
                                ref_top = tt.get('top', 0) or 0

                        for g in geom_tokens:
                            gtxt = re.sub(r"[^0-9]", "", g.get('text') or "")
                            if not re.fullmatch(r"\d", gtxt):
                                continue
                            gconf = (g.get('conf') or 0)
                            if gconf < 30:
                                continue
                            g_cx = (g.get('left', 0) or 0) + ((g.get('w', g.get('width', 0)) or 0) / 2)
                            g_top = g.get('top', 0) or 0
                            horiz_pen = abs(g_cx - ref_left) if ref_left is not None else 0
                            vert_pen = abs(g_top - ref_top) if ref_top is not None else 0
                            base_score = gconf - horiz_pen * 0.02 - vert_pen * 0.01

                            # try PREPEND
                            cand_pre = gtxt + cur_int
                            if not re.match(r"^0\d+", cand_pre) and len(cand_pre) <= 4:
                                score_pre = base_score + len(cand_pre) * 2
                                if score_pre > best_variant_score:
                                    best_variant_score = score_pre
                                    best_variant = (cand_pre, f"{cand_pre},{cur_dec}", "prepend", base_score)

                            # try APPEND
                            cand_app = cur_int + gtxt
                            if not re.match(r"^0\d+", cand_app) and len(cand_app) <= 4:
                                score_app = base_score + len(cand_app) * 2
                                if score_app > best_variant_score:
                                    best_variant_score = score_app
                                    best_variant = (cand_app, f"{cand_app},{cur_dec}", "append", base_score)

                        if best_variant:
                            new_int, combined, mode, base_score = best_variant
                            try:
                                new_val = float(_norm_num(combined))
                            except Exception:
                                new_val = None
                            if new_val is not None:
                                monto_val = new_val
                                monto_raw = combined
                                monto_debug.append({"chosen": monto_raw, "parsed": monto_val, "method": f"aggressive_global_{mode}"})
        except Exception:
            # be conservative: don't crash analysis on this best-effort heuristic
            pass
        try:
            alpha = [tk for tk in tokens if any(c.isalpha() for c in (tk.get("text") or "")) and (tk.get("conf") or 0) >= 35]
            alpha = [tk for tk in alpha if not looks_fiscal(tk.get("text") or "")]
            if alpha:
                alpha.sort(key=lambda x: (x.get("top", 0), x.get("left", 0)))
                clusters = []
                for tk in alpha:
                    if not clusters: clusters.append([tk]); continue
                    last = clusters[-1][-1]
                    if abs(tk.get("top", 0) - last.get("top", 0)) <= 30: clusters[-1].append(tk)
                    else: clusters.append([tk])
                best = None; best_score = -1
                for cl in clusters[:6]:
                    name = " ".join(t.get("text", "") for t in cl).strip(' ,.-:;')
                    letters = sum(1 for c in name if c.isalpha())
                    top_avg = sum(t.get("top", 0) for t in cl) / len(cl)
                    s = letters * 3 - (top_avg / 50)
                    if s > best_score and letters >= 3: best_score = s; best = name
                if best:
                    mclean = re.sub(r"\b(S\.A\.?|SA\.?|S\.A|CIF:|CIF|TEL:?).*$", "", best, flags=re.IGNORECASE).strip(' ,.-:;')
                    if sum(1 for c in mclean if c.isalpha()) >= 3 and not any(ch.isdigit() for ch in mclean): merchant = mclean
        except Exception:
            merchant = None

    if not merchant:
        for ln in texto_lines[:6]:
            letters = sum(1 for c in ln if c.isalpha()); digits = sum(1 for c in ln if c.isdigit())
            if letters > digits and letters > 3 and not looks_fiscal(ln):
                merchant = ln; break

    result: Dict[str, Any] = {"fecha": fecha, "monto": monto_val, "texto": original_text, "texto_clean": texto_clean, "texto_lines": texto_lines}
    if merchant: result["merchant"] = merchant
    if monto_raw: result["monto_raw"] = monto_raw
    if monto_debug: result["monto_debug"] = monto_debug
    if fecha_debug: result["fecha_debug"] = fecha_debug
    return result
