"""
CI script for GitHub Actions: import Ständerat (SR) votes from the official
parliament XLS files.

The OData API has no per-member SR roll-call data (see
scraper.scrape_sr_votes_direct), so SR votes come from a separate per-session
XLS file published at parlament.ch. This script seeds an ephemeral DB with
the already-committed members/sessions parquet files (so xls_importer can
match names and session dates), downloads any missing SR XLS files, imports
them, and merges the result into votes.parquet.

Run locally:  python ci_import_sr.py
"""

import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    from db import init_db, get_conn
    from scraper import _ensure_columns
    from xls_importer import download_sr_xls, import_xls_dir

    log("=== CI SR Import ===")
    init_db()
    _ensure_columns()

    mp = HERE / "members.parquet"
    sp = HERE / "sessions.parquet"
    if not mp.exists() or not sp.exists():
        log("ERROR: members.parquet / sessions.parquet not found")
        sys.exit(1)

    members_df = pd.read_parquet(mp)
    sessions_df = pd.read_parquet(sp)

    log(f"Seeding DB with {len(members_df)} members and {len(sessions_df)} sessions...")
    with get_conn() as conn:
        for _, row in members_df.iterrows():
            conn.execute(
                "INSERT OR IGNORE INTO members "
                "(PersonNumber, FirstName, LastName, CantonAbbr, CouncilAbbr, "
                " FractionAbbr, FractionName, PartyAbbr, PartyName) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (int(row["PersonNumber"]), str(row.get("FirstName", "")),
                 str(row.get("LastName", "")), str(row.get("CantonAbbr", "")),
                 str(row.get("CouncilAbbr", "")), str(row.get("FractionAbbr", "")),
                 str(row.get("FractionName", "")), str(row.get("PartyAbbr", "")),
                 str(row.get("PartyName", ""))),
            )
        for _, row in sessions_df.iterrows():
            conn.execute(
                "INSERT OR IGNORE INTO sessions (ID, Name, StartDate, EndDate, Legislature) "
                "VALUES (?,?,?,?,?)",
                (int(row["ID"]), str(row.get("Name", "")),
                 str(row.get("StartDate", ""))[:10], str(row.get("EndDate", ""))[:10], 52),
            )

    download_dir = HERE / "xls_downloads"
    log(f"Downloading SR XLS files to {download_dir}...")
    download_sr_xls(download_dir, log=log)

    log("Importing SR XLS files...")
    new_total = import_xls_dir(download_dir, council="SR", log=log)
    log(f"  {new_total} SR vote entries imported")

    if new_total == 0:
        log("No SR votes imported — nothing to merge.")
        return

    with get_conn() as conn:
        new_votes = pd.read_sql("SELECT * FROM votes", conn)

    vp = HERE / "votes.parquet"
    existing_votes = pd.read_parquet(vp)
    existing_votes["Decision"] = existing_votes["Decision"].astype(str)
    new_votes["Decision"] = new_votes["Decision"].astype(str)
    combined = (
        pd.concat([existing_votes, new_votes])
        .drop_duplicates(subset=["GroupKey", "PersonNumber"], keep="last")
        .reset_index(drop=True)
    )

    prev_count = len(existing_votes)
    if len(combined) < prev_count:
        log(f"  ERROR: combined row count ({len(combined):,}) is below "
            f"previous ({prev_count:,}). Aborting to avoid overwriting good data.")
        sys.exit(1)

    combined.to_parquet(vp, compression="zstd", index=False)
    log(f"votes.parquet: {len(combined):,} rows (+{len(combined) - prev_count:,})")
    log("=== Done ===")


if __name__ == "__main__":
    main()
