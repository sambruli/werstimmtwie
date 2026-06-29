"""
Import SR (and NR) voting data from official parliament XLS files.

The Swiss Parliament publishes individual roll-call vote records as XLS files
at https://www.parlament.ch/de/ratsbetrieb/abstimmungen/abstimmung-sr-xls
(one file per session, one row per roll call, one column per member).

Usage:
    python xls_importer.py                     # download + import all L52 SR sessions
    python xls_importer.py --dir /path/to/xls  # import from local directory
    python xls_importer.py --file foo.xlsx      # import a single file
"""

import argparse
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import openpyxl

from db import get_conn, init_db
from scraper import _ensure_columns, LEGISLATURE

# Parliament XLS files come in multiple format variants across legislatures.
# We detect the layout at parse time rather than hardcoding row/column constants.

# ── Vote value normalisation ──────────────────────────────────────────────────
_VOTE_MAP = {
    "ja": "Ja",
    "oui": "Ja",
    "yes": "Ja",
    "nein": "Nein",
    "non": "Nein",
    "no": "Nein",
    "enthaltung": "Enthaltung",
    "abstention": "Enthaltung",
    "entschuldigt": "Entschuldigt",
    "excusé": "Entschuldigt",
    "hat nicht teilgenommen": "Nicht teilgenommen",
    "n'a pas participé": "Nicht teilgenommen",
    "die präsidentin/der präsident stimmt nicht": "Präsident",
    "la présidente/le président ne vote pas": "Präsident",
    "abwesend": "Abwesend",
    "absent": "Abwesend",
}

def _norm_vote(raw):
    if raw is None:
        return "—"
    s = str(raw).strip()
    mapped = _VOTE_MAP.get(s.lower())
    if mapped:
        return mapped
    # Catch "Entschuldigt gemäss Art. 57 Abs. 4" and similar variants
    if s.lower().startswith("entschuldigt"):
        return "Entschuldigt"
    return s

def _parse_date(s):
    if not s:
        return ""
    try:
        return datetime.strptime(str(s).strip(), "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        return str(s)[:10]

def _biz_short(biz_num):
    """Format 8-digit business number as 'YYYY.NNN'."""
    if not biz_num:
        return ""
    try:
        s = str(int(biz_num))
    except (ValueError, TypeError):
        return str(biz_num)[:20]
    if len(s) == 8:
        return f"{s[:4]}.{s[4:]}"
    return s

# ── Session matching ──────────────────────────────────────────────────────────
def _find_session(dates, conn):
    """Return the session ID whose date range covers the most vote dates."""
    sessions = conn.execute(
        "SELECT ID, Name, StartDate, EndDate FROM sessions ORDER BY StartDate"
    ).fetchall()
    best_id, best_name, best_count = None, None, 0
    for s in sessions:
        sid, name, start, end = s["ID"], s["Name"], s["StartDate"], s["EndDate"]
        count = sum(1 for d in dates if d and start <= d <= end)
        if count > best_count:
            best_id, best_name, best_count = sid, name, count
    return best_id, best_name

# ── Member matching ───────────────────────────────────────────────────────────
def _build_member_lookup(conn, council="SR"):
    """Return dict 'Nachname, Vorname' (lowercase) → PersonNumber."""
    rows = conn.execute(
        "SELECT PersonNumber, FirstName, LastName FROM members WHERE CouncilAbbr=?",
        (council,)
    ).fetchall()
    lookup = {}
    for r in rows:
        key = f"{r['LastName']}, {r['FirstName']}".lower()
        lookup[key] = r["PersonNumber"]
    return lookup

def _fuzzy_match(name_xls, lookup):
    """Try exact match, then strip accents / punctuation variants."""
    key = name_xls.lower().strip()
    if key in lookup:
        return lookup[key]
    # normalise common accent differences (ü→ue, ä→ae, ö→oe, é→e, …)
    def _norm_str(s):
        return (s.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe")
                 .replace("é", "e").replace("è", "e").replace("ê", "e")
                 .replace("â", "a").replace("à", "a").replace("î", "i")
                 .replace("ô", "o").replace("û", "u").replace("ç", "c")
                 .replace("'", "'"))
    norm_key = _norm_str(key)
    for k, pn in lookup.items():
        if _norm_str(k) == norm_key:
            return pn
    return None

# ── Format auto-detection ─────────────────────────────────────────────────────

def _detect_layout(rows):
    """
    Scan the first 20 rows to find member names and column positions.
    Returns (member_cols, col_map, data_row_start) or raises ValueError.

    Handles all parliament XLS format variants (pre-2024, 2024, 2025+).
    """
    member_row_idx = None
    member_col_start = None
    header_row_idx = None

    for i, row in enumerate(rows[:20]):
        for j, cell in enumerate(row):
            cs = str(cell or "").lower()
            # Label cell before member names
            if "name des ratsmitglied" in cs:
                member_row_idx = i
                member_col_start = j + 1  # names start one col to the right of the label
                break
        if member_row_idx is not None:
            break

    if member_row_idx is None:
        raise ValueError("Kein Mitglieder-Header gefunden")

    # Find column-header row: the first row after member header whose col 0
    # contains a recognisable metadata keyword
    for i in range(member_row_idx + 1, min(len(rows), member_row_idx + 15)):
        row = rows[i]
        cs0 = str(row[0] if row else "").lower()
        if any(k in cs0 for k in ("geschäftsnummer", "abstimmungsdatum", "geschaftsnummer")):
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError("Kein Spaltenheader-Row gefunden")

    # Build column map from header row
    hrow = rows[header_row_idx]
    col_map = {}
    for j, cell in enumerate(hrow):
        cs = str(cell or "").lower()
        if "geschäftsnummer" in cs or "geschaftsnummer" in cs:
            col_map.setdefault("biz_num", j)
        if "geschäftstitel" in cs or "geschaftstitel" in cs:
            col_map.setdefault("biz_title", j)
        if "vorlagetitel" in cs or "entwurftitel" in cs:
            col_map.setdefault("draft_title", j)
        if "referenznummer" in cs:
            col_map["ref_num"] = j
        if "datum der abstimmung" in cs or "abstimmungsdatum" in cs:
            col_map["date"] = j
        if "abstimmungsgegenstand" in cs:
            col_map["subject"] = j
        if "bedeutung ja" in cs:
            col_map["mean_yes"] = j
        if "bedeutung nein" in cs:
            col_map["mean_no"] = j

    required = {"ref_num", "date"}
    missing = required - col_map.keys()
    if missing:
        raise ValueError(f"Fehlende Pflicht-Spalten: {missing}")

    # Collect member columns from the member name row
    name_row = rows[member_row_idx]
    member_cols = []
    for j in range(member_col_start, len(name_row)):
        name = name_row[j]
        if name and "," in str(name):
            member_cols.append({"col": j, "name_xls": str(name)})

    data_row_start = header_row_idx + 1
    return member_cols, col_map, data_row_start


# ── Core importer ─────────────────────────────────────────────────────────────
def import_xls(filepath, council="SR", log=None):
    """
    Import votes from one parliament XLS file into the DB.
    Returns number of vote rows stored.
    """
    _ensure_columns()
    path = Path(filepath)
    if not path.exists():
        if log:
            log(f"  Datei nicht gefunden: {filepath}")
        return 0

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # ── Auto-detect layout ────────────────────────────────────────────────
    try:
        member_cols, col_map, data_row_start = _detect_layout(rows)
    except ValueError as e:
        if log:
            log(f"  {path.name}: Layout-Erkennung fehlgeschlagen — {e}")
        return 0

    if not member_cols:
        if log:
            log(f"  {path.name}: Keine Mitgliederspalten gefunden.")
        return 0

    # ── Match to DB PersonNumbers ─────────────────────────────────────────
    conn = get_conn()
    lookup = _build_member_lookup(conn, council)
    unmatched = []
    for m in member_cols:
        m["pn"] = _fuzzy_match(m["name_xls"], lookup)
        if not m["pn"]:
            unmatched.append(m["name_xls"])

    if log:
        matched = len(member_cols) - len(unmatched)
        log(f"  {path.name}: {matched}/{len(member_cols)} Mitglieder zugeordnet")
        if unmatched:
            log(f"    Nicht gefunden: {unmatched[:5]}" +
                (f" (+{len(unmatched)-5} weitere)" if len(unmatched) > 5 else ""))

    # ── Find matching session from vote dates ────────────────────────────
    col_date = col_map["date"]
    col_ref  = col_map["ref_num"]
    col_biz  = col_map.get("biz_num")
    col_btit = col_map.get("biz_title")
    col_dtit = col_map.get("draft_title")
    col_subj = col_map.get("subject")
    col_yes  = col_map.get("mean_yes")
    col_no   = col_map.get("mean_no")

    def _cell(row, col):
        return row[col] if col is not None and col < len(row) else None

    data_rows = [r for r in rows[data_row_start:]
                 if len(r) > col_date and _cell(r, col_ref) is not None]
    dates = [_parse_date(_cell(r, col_date)) for r in data_rows]
    session_id, session_name = _find_session(dates, conn)

    if not session_id:
        if log:
            log(f"  {path.name}: Keine passende Session (Daten: {dates[:2]})")
        conn.close()
        return 0

    if log:
        log(f"  {path.name}: Session {session_id} ({session_name}), "
            f"{len(data_rows)} Abstimmungen")

    # ── Insert votes ──────────────────────────────────────────────────────
    total = 0
    for row in data_rows:
        ref_num   = _cell(row, col_ref)
        gk        = f"{council}-{ref_num}"
        date      = _parse_date(_cell(row, col_date))
        biz_num   = _cell(row, col_biz)
        biz_title = str(_cell(row, col_btit) or "")[:200]
        dft_title = str(_cell(row, col_dtit) or "")[:200]
        subject   = str(_cell(row, col_subj) or "")[:200]
        mean_yes  = str(_cell(row, col_yes)  or "")[:300]
        mean_no   = str(_cell(row, col_no)   or "")[:300]
        biz_short = _biz_short(biz_num)
        title     = biz_title or dft_title

        for m in member_cols:
            if not m["pn"]:
                continue
            raw  = _cell(row, m["col"])
            norm = _norm_vote(raw)
            conn.execute("""
                INSERT INTO votes
                (GroupKey, SessionId, PersonNumber, MeetingDate,
                 BizNumber, BizShort, BizTitle, Subject,
                 MeaningYes, MeaningNo, Decision, DecisionNorm)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(GroupKey, PersonNumber) DO UPDATE SET
                    MeaningYes = CASE WHEN excluded.MeaningYes != ''
                                 THEN excluded.MeaningYes ELSE MeaningYes END,
                    MeaningNo  = CASE WHEN excluded.MeaningNo  != ''
                                 THEN excluded.MeaningNo  ELSE MeaningNo  END
            """, (
                gk, session_id, m["pn"], date,
                str(biz_num or ""), biz_short, title, subject,
                mean_yes, mean_no, str(raw or ""), norm,
            ))
            total += 1

    conn.commit()
    conn.close()
    return total


def import_xls_dir(directory, council="SR", log=None):
    """Import all *_{council}_*.xlsx files found in a directory."""
    d = Path(directory)
    files = sorted(d.glob(f"*_{council}_*.xlsx")) + sorted(d.glob(f"*_{council}_*.xls"))
    if not files:
        if log:
            log(f"Keine {council}-XLS-Dateien in {directory} gefunden.")
        return 0
    total = 0
    for f in files:
        n = import_xls(f, council=council, log=log)
        total += n
        if log:
            log(f"  -> {n} Eintraege importiert")
    if log:
        log(f"Gesamt: {total} Einträge aus {len(files)} Dateien")
    return total


# ── Auto-download ─────────────────────────────────────────────────────────────
# URL pattern discovered from parlament.ch download links.
# Files follow: Abstimmungen_{NR|SR}_{YEAR}{CODE}_DE.xlsx
# where CODE: WS=Wintersession, FS=Frühjahrssession, SS=Sommersession,
#             HS=Herbstsession, SO=Sondersession (number varies)
_BASE_URL = (
    "https://www.parlament.ch/centers/documents/de/"
    "Abstimmungen_{council}_{code}_DE.xlsx"
)

# L52 session → file code mapping.
# Sondersessions: each entry lists the primary code to try plus alternates.
# The 4th/5th Sondersession of a year is numbered accordingly; exact parliament
# naming varies (SO4, SO5, etc.) so we try several candidates.
_L52_SESSION_CODES = {
    5201: ("SR", "2023WS"),
    5202: ("SR", "2024FS"),
    5203: ("SR", "2024SO4"),   # Sondersession 4. 2024
    5204: ("SR", "2024SS"),
    5205: ("SR", "2024HS"),
    5206: ("SR", "2024WS"),
    5207: ("SR", "2025FS"),
    5208: ("SR", "2025SO5"),   # Sondersession 5. 2025
    5209: ("SR", "2025SS"),
    5210: ("SR", "2025HS"),
    5211: ("SR", "2025WS"),
    5212: ("SR", "2026FS"),
    5213: ("SR", "2026SO4"),   # Sondersession 4. 2026
    5214: ("SR", "2026SS"),
}


def _try_download(url, dest, log=None):
    """Attempt to download url → dest. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                dest.write_bytes(resp.read())
                return True
    except Exception as e:
        if log:
            log(f"    Fehler: {e}")
    return False


def download_sr_xls(dest_dir, log=None):
    """
    Try to download all L52 SR session XLS files to dest_dir.
    Falls back to alternative URL patterns for Sondersessions.
    Returns list of successfully downloaded file paths.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for sid, (council, code) in _L52_SESSION_CODES.items():
        fname = f"Abstimmungen_{council}_{code}_DE.xlsx"
        fpath = dest / fname
        if fpath.exists():
            if log:
                log(f"  {fname}: bereits vorhanden")
            downloaded.append(fpath)
            continue

        url = _BASE_URL.format(council=council, code=code)
        if log:
            log(f"  {fname}: lade…")
        if _try_download(url, fpath, log):
            downloaded.append(fpath)
        else:
            # Try alternate codes for Sondersessions (different numbering conventions)
            year = code[:4]
            num_suffix = code[6:] if len(code) > 6 else ""
            alts = []
            for variant in ["SO", "SO1", "SO2", "SO3", "SO4", "SO5", "SO6"]:
                candidate = f"{year}{variant}"
                if candidate != code:
                    alts.append(candidate)
            for alt in alts[:5]:  # try up to 5 variants
                alt_url = _BASE_URL.format(council=council, code=alt)
                if log:
                    log(f"    Alternativ: {alt}…")
                alt_path = dest / f"Abstimmungen_{council}_{alt}_DE.xlsx"
                if _try_download(alt_url, alt_path, log):
                    downloaded.append(alt_path)
                    break
            else:
                if log:
                    log(f"    {fname}: nicht gefunden — bitte manuell herunterladen")

    return downloaded


# ── CLI ───────────────────────────────────────────────────────────────────────
def _run(args=None):
    ap = argparse.ArgumentParser(description="Import SR/NR XLS votes into DB")
    ap.add_argument("--file",    help="Single XLS file to import")
    ap.add_argument("--dir",     help="Directory of XLS files to import")
    ap.add_argument("--council", default="SR", choices=["SR", "NR"],
                    help="Which council's files to import (default: SR)")
    ap.add_argument("--download-dir", default="xls_downloads",
                    help="Directory for auto-downloaded files (default: xls_downloads)")
    ap.add_argument("--no-download", action="store_true",
                    help="Skip auto-download, only process local files")
    ns = ap.parse_args(args)

    init_db()
    log = print

    if ns.file:
        n = import_xls(ns.file, council=ns.council, log=log)
        print(f"Importiert: {n} Einträge")
    else:
        src_dir = ns.dir or ns.download_dir
        if not ns.no_download and ns.council == "SR":
            log(f"Lade SR-XLS-Dateien nach '{src_dir}'…")
            download_sr_xls(src_dir, log=log)
        n = import_xls_dir(src_dir, council=ns.council, log=log)
        print(f"Gesamt importiert: {n} Einträge")

    # Re-export parquet
    try:
        import export
        log("\nExportiere parquet…")
        export.export_all(log=log)
    except Exception as e:
        log(f"Export-Fehler: {e} — bitte manuell 'python export.py' ausführen")


if __name__ == "__main__":
    _run()
