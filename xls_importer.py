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

# ── XLS layout constants ──────────────────────────────────────────────────────
_ROW_TITLE    = 1   # 0-indexed row with session title
_ROW_NAMES    = 3   # member names ("Nachname, Vorname")
_ROW_CANTONS  = 4
_ROW_GENDERS  = 5
_ROW_FRACS    = 6
_ROW_HEADERS  = 7   # column header row (Geschäftsnummer, …)
_ROW_DATA     = 8   # first data row (0-indexed)
_COL_META_END = 10  # cols 0–9 are roll-call metadata; col 10 is separator
_COL_BIZ_NUM  = 0
_COL_BIZ_TITLE = 1
_COL_BIZ_TYPE  = 2
_COL_DRAFT_NUM = 3
_COL_DRAFT_TITLE = 4
_COL_REF_NUM  = 5   # Referenznummer — unique roll-call ID
_COL_DATE     = 6
_COL_SUBJECT  = 7
_COL_MEAN_YES = 8
_COL_MEAN_NO  = 9

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
    return _VOTE_MAP.get(str(raw).strip().lower(), str(raw))

def _parse_date(s):
    if not s:
        return ""
    try:
        return datetime.strptime(str(s).strip(), "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        return str(s)[:10]

def _biz_short(biz_num):
    """Format 8-digit business number as 'YYYY.NNN'."""
    s = str(int(biz_num)) if biz_num else ""
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

    # ── Parse member header rows ──────────────────────────────────────────
    names_row = rows[_ROW_NAMES]
    frac_row  = rows[_ROW_FRACS]

    member_cols = []
    for col_idx, name in enumerate(names_row):
        if name and "," in str(name) and col_idx >= _COL_META_END:
            frac_raw = frac_row[col_idx] if col_idx < len(frac_row) else ""
            frac = str(frac_raw or "").replace("Fraktion ", "").strip()
            member_cols.append({"col": col_idx, "name_xls": str(name), "frac": frac})

    if not member_cols:
        if log:
            log(f"  {path.name}: Keine Mitgliederspalten gefunden — Format unbekannt.")
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
                (f" … (+{len(unmatched)-5})" if len(unmatched) > 5 else ""))

    # ── Find matching session ─────────────────────────────────────────────
    data_rows = [r for r in rows[_ROW_DATA:]
                 if len(r) > _COL_BIZ_NUM and r[_COL_BIZ_NUM] is not None]
    dates = [_parse_date(r[_COL_DATE]) for r in data_rows]
    session_id, session_name = _find_session(dates, conn)

    if not session_id:
        if log:
            log(f"  {path.name}: Keine passende Session gefunden "
                f"(erste Daten: {dates[:3]})")
        conn.close()
        return 0

    if log:
        log(f"  {path.name}: Session {session_id} ({session_name}), "
            f"{len(data_rows)} Abstimmungen")

    # ── Insert votes ──────────────────────────────────────────────────────
    total = 0
    for row in data_rows:
        ref_num  = row[_COL_REF_NUM]
        gk       = f"{council}-{ref_num}"
        date     = _parse_date(row[_COL_DATE])
        biz_num  = row[_COL_BIZ_NUM]
        subject  = str(row[_COL_SUBJECT]  or "")[:200]
        mean_yes = str(row[_COL_MEAN_YES] or "")[:300]
        mean_no  = str(row[_COL_MEAN_NO]  or "")[:300]
        biz_title = str(row[_COL_BIZ_TITLE] or "")[:200]
        draft_title = str(row[_COL_DRAFT_TITLE] or "")[:200]
        biz_short = _biz_short(biz_num)
        # Use BizTitle if present, fall back to draft title
        title = biz_title or draft_title

        for m in member_cols:
            if not m["pn"]:
                continue
            raw  = row[m["col"]] if m["col"] < len(row) else None
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
            log(f"  → {n} Einträge importiert")
    if log:
        log(f"Gesamt: {total} Einträge aus {len(files)} Dateien")
    return total


# ── Auto-download ─────────────────────────────────────────────────────────────
# URL pattern discovered from parlament.ch download links.
# Files follow: Abstimmungen_{NR|SR}_{YEAR}{CODE}_DE.xlsx
# where CODE: WS=Wintersession, FS=Frühjahrssession, SS=Sommersession,
#             HS=Herbstsession, SO=Sondersession (number varies)
_BASE_URL = (
    "https://www.parlament.ch/de/ratsbetrieb/abstimmungen/"
    "abstimmung-{council_lc}-xls/-/media/Abstimmungen_{council}_{code}_DE.xlsx"
)

# L52 session → file code mapping
_L52_SESSION_CODES = {
    5201: ("SR", "2023WS"),
    5202: ("SR", "2024FS"),
    5203: ("SR", "2024SO1"),   # Sondersession – exact code may differ
    5204: ("SR", "2024SS"),
    5205: ("SR", "2024HS"),
    5206: ("SR", "2024WS"),
    5207: ("SR", "2025FS"),
    5208: ("SR", "2025SO1"),   # Sondersession – exact code may differ
    5209: ("SR", "2025SS"),
    5210: ("SR", "2025HS"),
    5211: ("SR", "2025WS"),
    5212: ("SR", "2026FS"),
    5213: ("SR", "2026SO1"),   # Sondersession – exact code may differ
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

        url = _BASE_URL.format(council_lc=council.lower(), council=council, code=code)
        if log:
            log(f"  {fname}: lade…")
        if _try_download(url, fpath, log):
            downloaded.append(fpath)
        else:
            # Try alternate Sondersession codes (SO, SO2, SS2, etc.)
            for alt in [code.replace("SO1", "SO"), code.replace("SO1", "SS2"),
                        code.replace("SO1", "SO2")]:
                if alt == code:
                    continue
                alt_url = _BASE_URL.format(
                    council_lc=council.lower(), council=council, code=alt)
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
