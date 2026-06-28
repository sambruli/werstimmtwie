import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "swiss_parl.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS members (
    PersonNumber   INTEGER PRIMARY KEY,
    FirstName      TEXT,
    LastName       TEXT,
    CantonAbbr     TEXT,
    CouncilAbbr    TEXT,
    FractionAbbr   TEXT,
    FractionName   TEXT,
    PartyAbbr      TEXT,
    PartyName      TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    ID             INTEGER PRIMARY KEY,
    Name           TEXT,
    StartDate      TEXT,
    EndDate        TEXT,
    Legislature    INTEGER,
    Scraped        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS votes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    GroupKey       TEXT,
    SessionId      INTEGER,
    PersonNumber   INTEGER,
    MeetingDate    TEXT,
    BizNumber      TEXT,
    BizShort       TEXT,
    BizTitle       TEXT,
    Subject        TEXT,
    TagName        TEXT DEFAULT '',
    MeaningYes     TEXT DEFAULT '',
    MeaningNo      TEXT DEFAULT '',
    Decision       INTEGER,
    DecisionNorm   TEXT,
    UNIQUE(GroupKey, PersonNumber)
);

CREATE INDEX IF NOT EXISTS idx_v_person  ON votes(PersonNumber);
CREATE INDEX IF NOT EXISTS idx_v_session ON votes(SessionId);
CREATE INDEX IF NOT EXISTS idx_v_group   ON votes(GroupKey);
CREATE INDEX IF NOT EXISTS idx_v_date    ON votes(MeetingDate);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(_SCHEMA)

def db_stats():
    with get_conn() as conn:
        members  = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        scraped  = conn.execute("SELECT COUNT(*) FROM sessions WHERE Scraped=1").fetchone()[0]
        votes    = conn.execute("SELECT COUNT(*) FROM votes").fetchone()[0]
    return {"members": members, "sessions": sessions, "scraped": scraped, "votes": votes}
