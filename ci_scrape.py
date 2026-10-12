"""
CI scraping script for GitHub Actions.

Workflow:
  1. Seed DB with existing session IDs (marked Scraped=1) so only new sessions are fetched
  2. Scrape current members + discover new sessions via API
  3. Scrape votes for new sessions only, merge with existing votes.parquet
  4. Export updated members.parquet and sessions.parquet
  5. Refresh biz_tags.parquet via bulk Curia Vista query

Run locally:  python ci_scrape.py
"""

import os
import sys
from datetime import date
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    from db import init_db, get_conn
    from scraper import (
        scrape_members, scrape_sessions, scrape_votes_session,
        backfill_biz_tags, migrate_decision_6, _ensure_columns,
    )

    log("=== CI Scrape ===")
    init_db()
    _ensure_columns()

    # Seed sessions table with already-scraped session IDs so update_scrape
    # only fetches sessions that appeared since the last run. A session whose
    # EndDate hasn't passed yet is seeded as Scraped=0 so its votes keep getting
    # re-fetched on every run until the session actually concludes.
    # FORCE_SESSION_IDS (comma-separated session IDs) overrides this for a
    # one-off manual backfill, e.g. when a past session's data turned out
    # incomplete despite its EndDate having already passed.
    force_ids = {
        int(s) for s in os.environ.get("FORCE_SESSION_IDS", "").split(",") if s.strip()
    }
    today = date.today().isoformat()
    sp = HERE / "sessions.parquet"
    existing_session_ids: set = set()
    if sp.exists():
        existing_df = pd.read_parquet(sp)
        existing_session_ids = set(existing_df["ID"])
        log(f"Seeding DB with {len(existing_session_ids)} existing sessions...")
        if force_ids:
            log(f"  Forcing re-scrape of session(s): {sorted(force_ids)}")
        with get_conn() as conn:
            for _, row in existing_df.iterrows():
                sid = int(row["ID"])
                end_date = str(row.get("EndDate", ""))[:10]
                scraped = 0 if sid in force_ids else (1 if end_date and end_date < today else 0)
                conn.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(ID, Name, StartDate, EndDate, Legislature, Scraped) VALUES (?,?,?,?,?,?)",
                    (sid, str(row.get("Name", "")),
                     str(row.get("StartDate", ""))[:10], end_date, 52, scraped),
                )

    # Scrape current active members and discover sessions from the API.
    log("Scraping members...")
    scrape_members(log)
    log("Scraping sessions...")
    scrape_sessions(log)

    # Identify sessions that are new (Scraped=0).
    with get_conn() as conn:
        pending = conn.execute(
            "SELECT ID, Name FROM sessions WHERE Scraped=0 ORDER BY StartDate"
        ).fetchall()
        all_pns = [r["PersonNumber"]
                   for r in conn.execute("SELECT PersonNumber FROM members").fetchall()]

    if not pending:
        log("No new sessions — skipping vote scrape.")
    else:
        log(f"{len(pending)} new session(s) to scrape ({len(all_pns)} members)...")
        for s in pending:
            log(f"  {s['Name']}...")
            scrape_votes_session(s["ID"], s["Name"], all_pns, log)
        migrate_decision_6(log)

        # Pull newly scraped votes from DB and merge with existing parquet.
        with get_conn() as conn:
            new_votes = pd.read_sql("SELECT * FROM votes", conn)
        log(f"  {len(new_votes):,} new votes scraped")

        vp = HERE / "votes.parquet"
        if vp.exists() and len(new_votes) > 0:
            existing_votes = pd.read_parquet(vp)
            existing_votes["Decision"] = existing_votes["Decision"].astype(str)
            new_votes["Decision"] = new_votes["Decision"].astype(str)
            combined = (
                pd.concat([existing_votes, new_votes])
                .drop_duplicates(subset=["GroupKey", "PersonNumber"], keep="last")
                .reset_index(drop=True)
            )
        else:
            combined = new_votes
            if "Decision" in combined.columns:
                combined["Decision"] = combined["Decision"].astype(str)

        if vp.exists():
            prev_count = len(pd.read_parquet(vp, columns=["GroupKey"]))
            if len(combined) < prev_count * 0.90:
                log(f"  ERROR: combined row count ({len(combined):,}) is >10% below "
                    f"previous ({prev_count:,}). Aborting to avoid overwriting good data.")
                sys.exit(1)

        combined.to_parquet(vp, compression="zstd", index=False)
        log(f"  votes.parquet: {len(combined):,} rows")

        # Export updated members and sessions parquets.
        with get_conn() as conn:
            members_df  = pd.read_sql(
                "SELECT * FROM members ORDER BY LastName, FirstName", conn)
            sessions_df = pd.read_sql(
                "SELECT ID, Name, StartDate, EndDate FROM sessions ORDER BY StartDate", conn)

        members_df.to_parquet(HERE / "members.parquet", compression="zstd", index=False)
        sessions_df.to_parquet(HERE / "sessions.parquet", compression="zstd", index=False)
        log(f"  members.parquet: {len(members_df)} rows")
        log(f"  sessions.parquet: {len(sessions_df)} rows")

    # Always refresh topic tags (two bulk API calls, ~30 s).
    log("Refreshing topic tags...")
    backfill_biz_tags(log)
    log("=== Done ===")


if __name__ == "__main__":
    main()
