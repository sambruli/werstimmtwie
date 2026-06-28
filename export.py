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

def export():
    if not DB.exists():
        print(f"ERROR: {DB} not found")
        return

    conn = sqlite3.connect(DB)

    tables = {
        "votes":   "SELECT * FROM votes",
        "members": "SELECT * FROM members ORDER BY LastName, FirstName",
        "sessions": "SELECT ID, Name, StartDate, EndDate FROM sessions ORDER BY StartDate",
    }

    for name, sql in tables.items():
        df = pd.read_sql(sql, conn)
        out = HERE / f"{name}.parquet"
        df.to_parquet(out, compression="zstd", index=False)
        mb = out.stat().st_size / 1e6
        print(f"  {name}.parquet: {len(df):,} rows — {mb:.1f} MB")

    conn.close()
    print("Done. Commit votes.parquet, members.parquet, sessions.parquet to git.")

if __name__ == "__main__":
    export()
