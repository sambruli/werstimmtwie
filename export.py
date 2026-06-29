"""
Export SQLite DB → Parquet files for deployment.
Run this after scraping new data, then commit votes.parquet + members.parquet + sessions.parquet.

Usage:
    .venv/Scripts/python.exe export.py
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB   = Path(__file__).parent / "swiss_parl.db"
HERE = Path(__file__).parent

def export(log=None):
    if log is None:
        log = print
    if not DB.exists():
        log(f"ERROR: {DB} not found")
        return

    conn = sqlite3.connect(DB)

    tables = {
        "votes":   "SELECT * FROM votes",
        "members": "SELECT * FROM members ORDER BY LastName, FirstName",
        "sessions": "SELECT ID, Name, StartDate, EndDate FROM sessions ORDER BY StartDate",
    }

    for name, sql in tables.items():
        df = pd.read_sql(sql, conn)
        # Ensure mixed-type columns are stored as strings (e.g. Decision can be int or str)
        if "Decision" in df.columns:
            df["Decision"] = df["Decision"].astype(str)
        out = HERE / f"{name}.parquet"
        df.to_parquet(out, compression="zstd", index=False)
        mb = out.stat().st_size / 1e6
        log(f"  {name}.parquet: {len(df):,} rows — {mb:.1f} MB")

    conn.close()
    log("Done. Commit votes.parquet, members.parquet, sessions.parquet to git.")

# alias used by xls_importer
export_all = export

if __name__ == "__main__":
    export()
