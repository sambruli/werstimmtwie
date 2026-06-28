"""
Fetches data from the Swiss Parliament OData API via swissparlpy.

Strategy: per-member per-session vote fetching to avoid server timeouts.
SR sessions are fetched separately since they differ from NR sessions.
"""

import time
import swissparlpy as spp
from db import get_conn, init_db

LEGISLATURE    = 52
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5    # seconds between retries
REQUEST_DELAY  = 0.25 # seconds between API calls

_DECISION_MAP = {
    "1": "Ja",           "2": "Nein",        "3": "Enthaltung",
    "4": "Abwesend",     "5": "Präsident",   "6": "Entschuldigt",
    "7": "Nicht teilgenommen",
    "ja": "Ja",          "oui": "Ja",        "yes": "Ja",
    "nein": "Nein",      "non": "Nein",      "no": "Nein",
    "enthaltung": "Enthaltung",  "abstention": "Enthaltung",
    "abwesend": "Abwesend",      "absent": "Abwesend",
    "entschuldigt": "Entschuldigt",
}

def _norm(raw):
    if raw is None:
        return "—"
    return _DECISION_MAP.get(str(raw).strip().lower(), str(raw))


def _fetch_with_retry(entity, log=None, **kwargs):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return list(spp.get_data(entity, **kwargs))
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                if log:
                    log(f"    Versuch {attempt} fehlgeschlagen ({e}), warte {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise


def scrape_members(log=None):
    conn = get_conn()
    total = 0
    for abbr in ("NR", "SR"):
        try:
            members = _fetch_with_retry(
                "MemberCouncil", log,
                Language="DE", CouncilAbbreviation=abbr,
            )
        except Exception as e:
            if log:
                log(f"  {abbr}: Fehler – {e}")
            continue

        for m in members:
            pn = m.get("PersonNumber")
            if not pn or not m.get("Active"):
                continue
            conn.execute("""
                INSERT OR REPLACE INTO members
                (PersonNumber,FirstName,LastName,CantonAbbr,CouncilAbbr,
                 FractionAbbr,FractionName,PartyAbbr,PartyName)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                pn, m.get("FirstName"), m.get("LastName"),
                m.get("CantonAbbreviation"), abbr,
                m.get("ParlGroupAbbreviation"), m.get("ParlGroupName"),
                m.get("PartyAbbreviation"), m.get("PartyName"),
            ))
            total += 1
        conn.commit()
        if log:
            log(f"  {abbr}: {total} Mitglieder gesamt")
    conn.close()
    return total


def scrape_sessions(log=None):
    """Fetch sessions for both NR and SR (they may differ)."""
    conn = get_conn()
    count = 0
    # Try fetching with and without council filter; the API may return both
    for council_id in (None, 1, 2):
        try:
            kwargs = dict(Language="DE", LegislativePeriodNumber=LEGISLATURE)
            if council_id is not None:
                kwargs["CouncilId"] = council_id
            sessions = _fetch_with_retry("Session", log, **kwargs)
        except Exception:
            continue

        for s in sessions:
            sid = s.get("ID")
            if not sid:
                continue
            name = (s.get("Name") or s.get("SessionName") or
                    s.get("Title") or f"Session {sid}")
            conn.execute("""
                INSERT OR IGNORE INTO sessions (ID,Name,StartDate,EndDate,Legislature)
                VALUES (?,?,?,?,?)
            """, (
                sid, name,
                str(s.get("StartDate") or "")[:10],
                str(s.get("EndDate") or "")[:10],
                LEGISLATURE,
            ))
            count += 1
        conn.commit()

    conn.close()
    if log:
        log(f"  {count} Sessions gefunden (inkl. Duplikate durch mehrfache Abfragen)")
    return count


def _group_key(v, session_id):
    ivid = v.get("IdVoting") or v.get("VotingId")
    if ivid:
        return str(ivid)
    date  = str(v.get("MeetingDate") or "")[:10]
    short = str(v.get("BusinessShortNumber") or v.get("BusinessNumber") or "")
    subj  = str(v.get("Subject") or "")[:40]
    return f"{session_id}|{date}|{short}|{subj}"


def _store_vote(conn, v, session_id):
    gk  = _group_key(v, session_id)
    dec = v.get("Decision")
    title       = str(v.get("BusinessTitle") or v.get("BillTitle") or v.get("Subject") or "")[:200]
    tag         = str(v.get("TagName")    or v.get("Tag")     or "")[:200]
    meaning_yes = str(v.get("MeaningYes") or "")[:300]
    meaning_no  = str(v.get("MeaningNo")  or "")[:300]
    conn.execute("""
        INSERT INTO votes
        (GroupKey,SessionId,PersonNumber,MeetingDate,
         BizNumber,BizShort,BizTitle,Subject,TagName,
         MeaningYes,MeaningNo,Decision,DecisionNorm)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(GroupKey,PersonNumber) DO UPDATE SET
            MeaningYes = CASE WHEN excluded.MeaningYes != '' THEN excluded.MeaningYes ELSE MeaningYes END,
            MeaningNo  = CASE WHEN excluded.MeaningNo  != '' THEN excluded.MeaningNo  ELSE MeaningNo  END
    """, (
        gk, session_id, v.get("PersonNumber"),
        str(v.get("MeetingDate") or "")[:10],
        str(v.get("BusinessNumber") or ""),
        str(v.get("BusinessShortNumber") or ""),
        title,
        str(v.get("Subject") or "")[:200],
        tag,
        meaning_yes,
        meaning_no,
        dec,
        _norm(dec),
    ))


def scrape_votes_session(session_id, session_name, member_pns, log=None):
    conn = get_conn()
    row = conn.execute("SELECT Scraped FROM sessions WHERE ID=?",
                       (session_id,)).fetchone()
    if row and row["Scraped"]:
        conn.close()
        if log:
            log(f"  {session_name}: bereits gescrapt")
        return 0

    if log:
        log(f"  {session_name}: lade {len(member_pns)} Mitglieder...")

    total, failed = 0, 0
    for i, pn in enumerate(member_pns):
        time.sleep(REQUEST_DELAY)
        try:
            votes = _fetch_with_retry(
                "Voting", log,
                Language="DE", IdSession=session_id, PersonNumber=pn,
            )
            for v in votes:
                _store_vote(conn, v, session_id)
                total += 1
        except Exception as e:
            failed += 1
            if log and failed <= 3:
                log(f"    PersonNumber {pn}: {e}")

        if (i + 1) % 50 == 0:
            conn.commit()
            if log:
                log(f"    {i+1}/{len(member_pns)} Mitglieder, {total} Einträge...")

    conn.commit()
    conn.execute("UPDATE sessions SET Scraped=1 WHERE ID=?", (session_id,))
    conn.commit()
    conn.close()
    if log:
        msg = f"  {session_name}: {total} Einträge"
        if failed:
            msg += f" ({failed} fehlgeschlagen)"
        log(msg)
    return total


def scrape_sr_votes_direct(log=None):
    """Fetch SR votes by PersonNumber only — no IdSession filter.
    The Voting API returns no SR data when filtered by NR session IDs."""
    _ensure_columns()
    conn = get_conn()
    sr_pns = [r["PersonNumber"] for r in
              conn.execute(
                  "SELECT PersonNumber FROM members WHERE CouncilAbbr='SR'"
              ).fetchall()]
    conn.close()

    if not sr_pns:
        if log:
            log("  SR: keine Mitglieder in DB")
        return 0

    if log:
        log(f"  SR: Lade Abstimmungen für {len(sr_pns)} Mitglieder (direkt)…")

    conn = get_conn()
    total, failed = 0, 0
    for i, pn in enumerate(sr_pns):
        time.sleep(REQUEST_DELAY)
        try:
            votes = _fetch_with_retry(
                "Voting", log,
                Language="DE", PersonNumber=pn,
            )
            for v in votes:
                # Get session ID from the vote itself
                sid = (v.get("IdSession") or v.get("SessionId")
                       or v.get("Session") or 0)
                try:
                    sid = int(sid)
                except (TypeError, ValueError):
                    sid = 0
                _store_vote(conn, v, sid)
                total += 1
        except Exception as e:
            failed += 1
            if log and failed <= 5:
                log(f"    SR {pn}: {e}")

        if (i + 1) % 10 == 0:
            conn.commit()
            if log:
                log(f"    {i+1}/{len(sr_pns)} SR, {total} Einträge…")

    conn.commit()
    conn.close()
    if log:
        msg = f"  SR: {total} Abstimmungseinträge geladen"
        if failed:
            msg += f" ({failed} Fehler)"
        log(msg)
    return total


def backfill_meaning(log=None):
    """
    Re-scrape sessions that have votes with empty MeaningYes to fill in
    the MeaningYes / MeaningNo fields without touching other data.
    Uses the upsert in _store_vote to update only those two columns.
    """
    _ensure_columns()
    conn = get_conn()

    # Find sessions that have at least one vote with empty MeaningYes
    stale = conn.execute("""
        SELECT DISTINCT s.ID, s.Name
        FROM sessions s
        JOIN votes v ON v.SessionId = s.ID
        WHERE (v.MeaningYes IS NULL OR v.MeaningYes = '')
        ORDER BY s.StartDate
    """).fetchall()

    all_pns = [r["PersonNumber"] for r in
               conn.execute("SELECT PersonNumber FROM members").fetchall()]
    conn.close()

    if not stale:
        if log:
            log("Alle Sessions haben bereits Bedeutungsdaten — nichts zu tun.")
        return 0

    if log:
        log(f"{len(stale)} Sessions zum Nachfüllen gefunden…")

    # Temporarily reset Scraped flag for stale sessions so scrape_votes_session runs
    with get_conn() as conn:
        for s in stale:
            conn.execute("UPDATE sessions SET Scraped=0 WHERE ID=?", (s["ID"],))

    total = 0
    for i, s in enumerate(stale):
        if log:
            log(f"  {i+1}/{len(stale)}: {s['Name']}")
        total += scrape_votes_session(s["ID"], s["Name"], all_pns, log)

    # Also re-run SR direct scrape (no session filter needed, upsert handles it)
    if log:
        log("SR-Daten nachfüllen…")
    total += scrape_sr_votes_direct(log)

    if log:
        log(f"Backfill abgeschlossen: {total:,} Einträge aktualisiert.")
    return total


def migrate_decision_6(log=None):
    """One-time fix: update existing '6' DecisionNorm values in the DB."""
    with get_conn() as conn:
        n = conn.execute(
            "UPDATE votes SET DecisionNorm='Entschuldigt' WHERE DecisionNorm='6'"
        ).rowcount
        conn.execute(
            "UPDATE votes SET DecisionNorm='Nicht teilgenommen' WHERE DecisionNorm='7'"
        )
    if log and n > 0:
        log(f"  Migration: {n} Einträge von '6' → 'Entschuldigt' korrigiert")


def full_scrape(log=None):
    init_db()
    _ensure_columns()

    if log:
        log("Lade Mitglieder...")
    n_m = scrape_members(log)

    if log:
        log("Lade Sessions...")
    n_s = scrape_sessions(log)

    conn = get_conn()
    sessions = conn.execute(
        "SELECT ID, Name FROM sessions ORDER BY StartDate"
    ).fetchall()
    nr_pns = [r["PersonNumber"] for r in
              conn.execute("SELECT PersonNumber FROM members WHERE CouncilAbbr='NR'").fetchall()]
    sr_pns = [r["PersonNumber"] for r in
              conn.execute("SELECT PersonNumber FROM members WHERE CouncilAbbr='SR'").fetchall()]
    conn.close()

    if not nr_pns and not sr_pns:
        if log:
            log("FEHLER: Keine Mitglieder in der Datenbank.")
        return n_m, n_s, 0

    if log:
        log(f"Lade Abstimmungen ({len(sessions)} Sessions, "
            f"{len(nr_pns)} NR + {len(sr_pns)} SR Mitglieder)...")

    total_v = 0
    for i, s in enumerate(sessions):
        if log:
            log(f"Session {i+1}/{len(sessions)}: {s['Name']}")
        # Try all members; SR members will return 0 for NR sessions and vice versa
        all_pns = nr_pns + sr_pns
        total_v += scrape_votes_session(s["ID"], s["Name"], all_pns, log)

    migrate_decision_6(log)

    if log:
        log("Lade SR-Abstimmungen direkt (kein Session-Filter)…")
    total_v += scrape_sr_votes_direct(log)

    return n_m, n_s, total_v


def update_scrape(log=None):
    init_db()
    _ensure_columns()
    scrape_members(log)
    scrape_sessions(log)
    migrate_decision_6(log)

    conn = get_conn()
    pending = conn.execute(
        "SELECT ID, Name FROM sessions WHERE Scraped=0 ORDER BY StartDate"
    ).fetchall()
    all_pns = [r["PersonNumber"] for r in
               conn.execute("SELECT PersonNumber FROM members").fetchall()]
    conn.close()

    total_v = 0
    for s in pending:
        if log:
            log(f"Session: {s['Name']}")
        total_v += scrape_votes_session(s["ID"], s["Name"], all_pns, log)

    if log:
        log("Lade SR-Abstimmungen direkt…")
    total_v += scrape_sr_votes_direct(log)
    return total_v


def _ensure_columns():
    """Add any missing columns to existing DBs (idempotent)."""
    with get_conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(votes)").fetchall()}
        for col, defn in [
            ("TagName",    "TEXT DEFAULT ''"),
            ("MeaningYes", "TEXT DEFAULT ''"),
            ("MeaningNo",  "TEXT DEFAULT ''"),
        ]:
            if col not in cols:
                conn.execute(f"ALTER TABLE votes ADD COLUMN {col} {defn}")


def _ensure_tagname_column():
    """Kept for backwards compatibility — calls _ensure_columns."""
    _ensure_columns()
