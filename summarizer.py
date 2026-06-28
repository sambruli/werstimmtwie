"""Claude-powered text summaries and SQL generation."""

import os
import re
from pathlib import Path
import anthropic

_ENV_FILE = Path(__file__).parent / ".env"

def _load_env():
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 1500

def _client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY nicht gesetzt.")
    return anthropic.Anthropic(api_key=key)

_SYS = (
    "Du bist ein Politikanalyst für das Schweizer Parlament. "
    "Deine Antworten sind präzise, sachlich und auf Deutsch. "
    "Nutze keine Floskeln. Halte dich an die Daten."
)

def _ask(prompt: str) -> str:
    msg = _client().messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=_SYS,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def summarize_mp_profile(name: str, fraction: str, stats: dict,
                          topic_rows: list[dict]) -> str:
    lines = [
        f"Parlamentsmitglied: {name} ({fraction})",
        f"Gesamtabweichung von der Fraktion: {stats['pct']}% "
        f"({stats['deviated']} von {stats['total']} Abstimmungen)",
        "", "Abweichung nach Thema:",
    ]
    for r in topic_rows:
        lines.append(f"  - {r['Thema']}: {r['Abweichung (%)']}% ({r['Total']} Abstimmungen)")
    lines.append(
        "\nFasse in 4–6 Sätzen zusammen, was diese Zahlen über das Abstimmungsverhalten "
        "dieser Person aussagen. Gehe auf auffällige Themen ein."
    )
    return _ask("\n".join(lines))


def summarize_deviation_votes(name: str, fraction: str,
                               deviated_rows: list[dict]) -> str:
    lines = [
        f"{name} ({fraction}) – abweichende Abstimmungen von der Fraktionsmehrheit:", "",
    ]
    for r in deviated_rows[:25]:
        lines.append(
            f"  {str(r.get('MeetingDate',''))[:10]}  {r.get('BizShort','')}  "
            f"{str(r.get('BizTitle',''))[:70]}  "
            f"[MP: {r.get('DecisionNorm','')} / Fraktion: {r.get('FracMajority','')}]"
        )
    lines.append(
        "\nWas lässt sich aus diesen Abweichungen über die politische Positionierung "
        "dieser Person schliessen? Gibt es erkennbare Muster? Antworte in 4–6 Sätzen."
    )
    return _ask("\n".join(lines))


def summarize_comparison(name_a: str, name_b: str,
                          frac_a: str, frac_b: str,
                          agreement: dict, diverging_rows: list[dict]) -> str:
    lines = [
        f"Vergleich: {name_a} ({frac_a}) vs. {name_b} ({frac_b})",
        f"Übereinstimmung: {agreement['pct']}% "
        f"({agreement['agree']} von {agreement['total']} gemeinsamen Abstimmungen)",
        "", "Grösste Divergenzen:",
    ]
    for r in diverging_rows[:20]:
        lines.append(
            f"  {str(r.get('MeetingDate',''))[:10]}  {str(r.get('BizTitle',''))[:70]}  "
            f"[{name_a}: {r.get('Vote_A','')}, {name_b}: {r.get('Vote_B','')}]"
        )
    lines.append(
        "\nFasse in 4–6 Sätzen zusammen, worin sich diese zwei Personen "
        "politisch unterscheiden und wo sie übereinstimmen."
    )
    return _ask("\n".join(lines))


def summarize_topic(topic: str, breakdown_rows: list[dict]) -> str:
    lines = [
        f"Thema: {topic}",
        "Abstimmungsverhalten nach Fraktion:", "",
    ]
    for r in breakdown_rows:
        parts = [f"{k}: {v}" for k, v in r.items() if k != "FractionAbbr"]
        lines.append(f"  {r.get('FractionAbbr','')}: {', '.join(parts)}")
    lines.append(
        "\nWie haben die verschiedenen Fraktionen zu diesem Thema abgestimmt? "
        "Wer war dafür, wer dagegen? Gibt es Überraschungen? Antworte in 4–6 Sätzen."
    )
    return _ask("\n".join(lines))


def summarize_fraction_deviators(fraction: str, rows: list[dict]) -> str:
    lines = [
        f"Fraktion: {fraction} — Abweichungsraten der Mitglieder:", "",
    ]
    for r in rows[:20]:
        lines.append(
            f"  {r['Name']} ({r['Kanton']}): {r['Abweichung (%)']}% "
            f"({r['Abweichend']} von {r['Total']} Abstimmungen)"
        )
    lines.append(
        "\nWer weicht am stärksten von der Fraktionslinie ab? "
        "Gibt es erkennbare Muster (z.B. Kanton, Flügel)? Antworte in 4–6 Sätzen."
    )
    return _ask("\n".join(lines))


_DB_SCHEMA = """
SQLite-Datenbank mit folgenden Tabellen:

members:
  PersonNumber INTEGER PRIMARY KEY, FirstName TEXT, LastName TEXT
  CouncilAbbr TEXT ('NR' oder 'SR'), FractionAbbr TEXT (z.B. 'RL'=FDP, 'S'=SP, 'V'=SVP, 'G'=Grüne, 'GL'=GLP, 'M-E'=Mitte)
  FractionName TEXT, PartyAbbr TEXT, CantonAbbr TEXT

sessions:
  ID INTEGER PRIMARY KEY, Name TEXT (z.B. 'Wintersession 2023'), StartDate TEXT, EndDate TEXT

votes:
  GroupKey TEXT (identifiziert eine einzelne Abstimmung — gleich für alle Mitglieder)
  SessionId INTEGER, PersonNumber INTEGER, MeetingDate TEXT (YYYY-MM-DD)
  BizShort TEXT (Geschäftsnummer z.B. '23.456'), BizTitle TEXT
  DecisionNorm TEXT ('Ja', 'Nein', 'Enthaltung', 'Abwesend', 'Entschuldigt', 'Nicht teilgenommen')

Hinweis: Themenfilter (Klima, Umwelt etc.) sind nicht in der DB — nur in Python berechnet.
"""


def _gen_sql(question: str) -> str:
    """Internal: generate SQL from a question."""
    prompt = (
        f"{_DB_SCHEMA}\n"
        f"Generiere eine SQLite-SELECT-Abfrage für:\n{question}\n\n"
        "Nur SELECT, LIMIT 500. Antworte ausschliesslich mit SQL, kein Markdown."
    )
    sql = _client().messages.create(
        model=MODEL, max_tokens=600,
        system="SQLite-Experte. Nur gültiges SQL, kein Text drumherum.",
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def chat_answer(question: str, history: list[dict],
                db_path: str) -> tuple[str, "pd.DataFrame | None"]:
    """
    Full pipeline: question → SQL → execute → natural language answer.
    Returns (answer_text, result_dataframe_or_None).
    history: list of {"role": "user"/"assistant", "content": str}
    """
    import sqlite3, pandas as pd

    # Generate SQL
    try:
        sql = _gen_sql(question)
    except Exception as e:
        return f"Fehler beim Generieren der Abfrage: {e}", None

    # Safety check
    if not sql.strip().upper().startswith("SELECT"):
        return "Nur Leseanfragen sind erlaubt.", None

    # Execute
    try:
        db = Path(db_path)
        if db.exists() and db.stat().st_size > 0:
            conn = sqlite3.connect(db_path)
        else:
            # Build in-memory DB from parquet files (Streamlit Cloud / no local DB)
            conn = sqlite3.connect(":memory:")
            here = db.parent
            for table in ("votes", "members", "sessions"):
                p = here / f"{table}.parquet"
                if p.exists():
                    pd.read_parquet(p).to_sql(table, conn, if_exists="replace", index=False)
        df = pd.read_sql(sql, conn)
        conn.close()
    except Exception as e:
        return f"Datenbankfehler: {e}\n\nGeneriertes SQL:\n```sql\n{sql}\n```", None

    # Formulate natural language answer
    if df.empty:
        data_summary = "Die Abfrage ergab keine Ergebnisse."
    else:
        data_summary = df.to_string(max_rows=30, index=False)

    conv = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    conv.append({"role": "user", "content": (
        f"Frage: {question}\n\n"
        f"Datenbankresultat ({len(df)} Zeilen):\n{data_summary}\n\n"
        "Beantworte die Frage direkt und präzise auf Deutsch, basierend auf diesen Daten. "
        "Nenne konkrete Zahlen und Namen. Keine SQL-Details erwähnen."
    )})

    answer = _client().messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=_SYS,
        messages=conv,
    ).content[0].text

    return answer, (df if not df.empty else None)
