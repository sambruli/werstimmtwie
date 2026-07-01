"""
Analysis functions — SQL-first for performance, pandas for aggregation.
"""

from functools import lru_cache
from pathlib import Path
import pandas as pd
from topics import classify, any_topic

_HERE = Path(__file__).parent

def _has_db() -> bool:
    return (_HERE / "swiss_parl.db").exists()

def _get_conn():
    from db import get_conn
    return get_conn()

FRACTION_OVERRIDES: dict = {}  # PersonNumber → FractionAbbr, if needed

# Jositsch was SP until recently; treat him as SP for group analysis
_NAME_OVERRIDES = {
    "Jositsch": "S",  # Daniel Jositsch — historically SP fraction
}

# Decisions excluded from deviation analysis (not a meaningful political vote)
# "Nicht teilgenommen" = intentional non-participation (distinct from Enthaltung),
# but still not a vote against the fraction, so excluded.
# Raw API codes 6/7 included as safety net for un-migrated rows.
ABSENT = {
    "Abwesend", "Entschuldigt", "Nicht teilgenommen", "Präsident", "—", "6", "7"
}


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_members() -> pd.DataFrame:
    p = _HERE / "members.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    elif _has_db():
        with _get_conn() as conn:
            df = pd.read_sql("SELECT * FROM members ORDER BY LastName, FirstName", conn)
    else:
        return pd.DataFrame()
    for pn, frac in FRACTION_OVERRIDES.items():
        df.loc[df["PersonNumber"] == pn, "FractionAbbr"] = frac
    for name, frac in _NAME_OVERRIDES.items():
        df.loc[df["LastName"] == name, "FractionAbbr"] = frac
    return df


def load_all_votes() -> pd.DataFrame:
    p = _HERE / "votes.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    elif _has_db():
        with _get_conn() as conn:
            df = pd.read_sql("SELECT * FROM votes", conn)
    else:
        return pd.DataFrame()
    # Restrict to sessions in sessions.parquet (current legislature only).
    # votes.parquet contains historical data for long-serving members; keeping
    # only known sessions prevents old cross-legislature votes from poisoning
    # faction-alignment and deviation calculations.
    sp = _HERE / "sessions.parquet"
    if sp.exists():
        valid = set(pd.read_parquet(sp)["ID"])
        df = df[df["SessionId"].isin(valid)]
    # Normalize BizShort: NR uses "24.024", SR uses "24.24" for the same affair.
    # Strip leading zeros after the dot so both chambers group under one Geschäft.
    if "BizShort" in df.columns:
        df["BizShort"] = df["BizShort"].str.replace(
            r"^(\d+)\.0+(\d+)$", r"\1.\2", regex=True
        )
    return df


def load_votes_for_mp(person_number: int, all_votes: pd.DataFrame | None = None) -> pd.DataFrame:
    av = all_votes if all_votes is not None else load_all_votes()
    return av[av["PersonNumber"] == person_number].copy()


def load_votes_for_fraction(members: pd.DataFrame, fraction_abbr: str,
                             all_votes: pd.DataFrame | None = None) -> pd.DataFrame:
    pns = set(members.loc[members["FractionAbbr"] == fraction_abbr, "PersonNumber"])
    if not pns:
        return pd.DataFrame()
    av = all_votes if all_votes is not None else load_all_votes()
    return av[av["PersonNumber"].isin(pns)].copy()


def load_sessions() -> pd.DataFrame:
    p = _HERE / "sessions.parquet"
    if p.exists():
        return pd.read_parquet(p)
    if _has_db():
        with _get_conn() as conn:
            return pd.read_sql(
                "SELECT ID, Name, StartDate FROM sessions ORDER BY StartDate", conn
            )
    return pd.DataFrame()


# ── Topic tagging (Curia Vista official tags via biz_tags.parquet) ───────────

@lru_cache(maxsize=1)
def _biz_tag_lookup() -> dict:
    """Returns {biz_number_str: [tag1, tag2, ...]} loaded once per process."""
    p = _HERE / "biz_tags.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    lookup = {}
    for _, row in df.iterrows():
        bn = str(row.get("BizNumber") or "").strip()
        tags_str = str(row.get("TagNames") or "").strip()
        if bn:
            lookup[bn] = [t.strip() for t in tags_str.split("|") if t.strip()] if tags_str else []
    return lookup


def load_available_tags() -> list[str]:
    """Sorted list of Curia Vista tags present in biz_tags.parquet.
    Falls back to custom TOPICS keys if the parquet hasn't been scraped yet."""
    lookup = _biz_tag_lookup()
    if not lookup:
        from topics import TOPICS
        return sorted(TOPICS.keys())
    tags: set[str] = set()
    for tag_list in lookup.values():
        tags.update(tag_list)
    return sorted(tags)


def _topic_text(df: pd.DataFrame) -> pd.Series:
    """Keyword fallback: concatenate title/subject/tagname columns."""
    cols = []
    for col in ("BizTitle", "Subject", "TagName"):
        if col in df.columns:
            cols.append(df[col].fillna(""))
    return cols[0].str.cat(cols[1:], sep=" ") if len(cols) > 1 else cols[0] if cols else pd.Series("", index=df.index)


def tag_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each vote row with Curia Vista official topic tags.
    Falls back to keyword matching if biz_tags.parquet is not present."""
    df = df.copy()
    lookup = _biz_tag_lookup()
    if lookup and "BizNumber" in df.columns:
        df["Topics"]   = df["BizNumber"].map(lambda bn: lookup.get(str(bn or "").strip(), []))
        df["AnyTopic"] = df["Topics"].map(bool)
    else:
        text = _topic_text(df)
        df["Topics"]   = text.map(classify)
        df["AnyTopic"] = text.map(any_topic)
    return df


def filter_by_topic(df: pd.DataFrame, topic: str | None) -> pd.DataFrame:
    if topic is None or topic == "Alle":
        return df
    if "Topics" not in df.columns:
        df = tag_topics(df)
    return df[df["Topics"].map(lambda t: topic in t)]


def curia_vista_url(biz_short: str, biz_number: str) -> str:
    """Build a Curia Vista URL for a Geschäftsnummer."""
    base = "https://www.parlament.ch/de/ratsbetrieb/suche-curia-vista/geschaeft?AffairId="
    # biz_number may already be an 8-digit ID
    bn = str(biz_number or "").strip()
    if bn.isdigit() and len(bn) == 8:
        return base + bn
    # biz_short like "23.456" or "2023.456"
    bs = str(biz_short or "").strip()
    if "." in bs:
        parts = bs.split(".")
        year, num = parts[0].strip(), parts[1].strip()
        if len(year) == 2:
            year = "20" + year
        if year.isdigit() and num.isdigit():
            return base + year + num.zfill(4)
    return ""


# ── Fraction majority ─────────────────────────────────────────────────────────

def fraction_majority(all_votes: pd.DataFrame,
                      members: pd.DataFrame,
                      fraction_abbr: str) -> pd.Series:
    """Series indexed by GroupKey → majority DecisionNorm for that fraction."""
    frac_pns = set(members.loc[members["FractionAbbr"] == fraction_abbr, "PersonNumber"])
    frac_votes = all_votes[
        all_votes["PersonNumber"].isin(frac_pns) &
        ~all_votes["DecisionNorm"].isin(ABSENT)
    ]
    if frac_votes.empty:
        return pd.Series(dtype=str)
    return frac_votes.groupby("GroupKey")["DecisionNorm"].agg(lambda s: s.mode().iloc[0])


def fraction_majority_strength(all_votes: pd.DataFrame,
                                members: pd.DataFrame,
                                fraction_abbr: str) -> pd.Series:
    """
    Series indexed by GroupKey → 'Ja (87%)' style string showing the
    majority decision AND what share of active fraction members held it.
    """
    frac_pns = set(members.loc[members["FractionAbbr"] == fraction_abbr, "PersonNumber"])
    frac_votes = all_votes[
        all_votes["PersonNumber"].isin(frac_pns) &
        ~all_votes["DecisionNorm"].isin(ABSENT)
    ]
    if frac_votes.empty:
        return pd.Series(dtype=str)

    def _strength(s):
        counts = s.value_counts()
        maj = counts.index[0]
        pct = round(counts.iloc[0] / len(s) * 100)
        return f"{maj} ({pct}%)"

    return frac_votes.groupby("GroupKey")["DecisionNorm"].agg(_strength)


# ── Fraction positions matrix ────────────────────────────────────────────────

def compute_fraction_positions(all_votes: pd.DataFrame,
                                members: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute the majority vote of every fraction on every roll call.
    Returns DataFrame: index=GroupKey, columns=FractionAbbr, values=DecisionNorm.
    This is the foundation for all direction/alignment analyses.
    """
    active = all_votes[~all_votes["DecisionNorm"].isin(ABSENT)]
    merged = active.merge(members[["PersonNumber", "FractionAbbr"]], on="PersonNumber", how="inner")
    if merged.empty:
        return pd.DataFrame()
    return (
        merged.groupby(["GroupKey", "FractionAbbr"])["DecisionNorm"]
        .agg(lambda s: s.mode().iloc[0])
        .unstack()
    )


def deviation_alignment_fast(deviated_df: pd.DataFrame, own_frac: str,
                              positions: pd.DataFrame) -> pd.DataFrame:
    """
    For an MP's deviated votes, compute how often they aligned with each other fraction.
    Uses the pre-computed positions matrix — no need to load all_votes again.
    """
    if deviated_df.empty or positions.empty:
        return pd.DataFrame()
    mp_votes = deviated_df.set_index("GroupKey")["DecisionNorm"]
    results = []
    for frac in positions.columns:
        if frac == own_frac:
            continue
        frac_pos = positions[frac].dropna()
        common = mp_votes.index.intersection(frac_pos.index)
        if len(common) == 0:
            continue
        align = int((mp_votes[common] == frac_pos[common]).sum())
        if align == 0:
            continue
        results.append({
            "Fraktion":          frac,
            "Übereinstimmungen": align,
            "Anteil (%)":        round(align / len(deviated_df) * 100, 1),
        })
    return pd.DataFrame(results).sort_values("Übereinstimmungen", ascending=False)


def fraction_direction_heatmap(fraction_abbr: str, all_votes: pd.DataFrame,
                                members: pd.DataFrame,
                                positions: pd.DataFrame) -> pd.DataFrame:
    """
    For each MP in the fraction, what % of their deviations aligned with each other fraction?
    Returns DataFrame (MPs × other fractions) suitable for a heatmap.
    """
    frac_maj = fraction_majority(all_votes, members, fraction_abbr)
    frac_members = members[members["FractionAbbr"] == fraction_abbr]
    other_fracs = [c for c in positions.columns if c != fraction_abbr]

    rows = {}
    for _, m in frac_members.iterrows():
        pn = m["PersonNumber"]
        mp_v = all_votes[all_votes["PersonNumber"] == pn].copy()
        if mp_v.empty:
            continue
        mp_v["FracMaj"] = mp_v["GroupKey"].map(frac_maj)
        active = mp_v[~mp_v["DecisionNorm"].isin(ABSENT)]
        dev = active[
            active["DecisionNorm"] != active["FracMaj"]
        ].dropna(subset=["FracMaj"])
        if len(dev) < 5:
            continue
        mp_dev = dev.set_index("GroupKey")["DecisionNorm"]
        row = {}
        for frac in other_fracs:
            if frac not in positions.columns:
                continue
            frac_pos = positions[frac].dropna()
            common = mp_dev.index.intersection(frac_pos.index)
            if len(common) == 0:
                row[frac] = 0.0
                continue
            row[frac] = round((mp_dev[common] == frac_pos[common]).sum() / len(dev) * 100, 1)
        rows[f"{m['FirstName']} {m['LastName']}"] = row

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


# ── MP deviation from their fraction ─────────────────────────────────────────

def mp_deviation(mp_pn: int,
                 all_votes: pd.DataFrame,
                 members: pd.DataFrame) -> pd.DataFrame:
    """
    MP's votes enriched with FracMajority and Deviated columns.
    Loads MP votes from SQL if all_votes is not pre-filtered.
    """
    mp_row = members[members["PersonNumber"] == mp_pn]
    if mp_row.empty:
        return pd.DataFrame()
    frac = mp_row.iloc[0]["FractionAbbr"]

    frac_maj = fraction_majority(all_votes, members, frac)

    mp_votes = all_votes[all_votes["PersonNumber"] == mp_pn].copy()
    mp_votes["FracMajority"] = mp_votes["GroupKey"].map(frac_maj)
    active = ~mp_votes["DecisionNorm"].isin(ABSENT)
    mp_votes["Deviated"] = active & (mp_votes["DecisionNorm"] != mp_votes["FracMajority"])
    return mp_votes


def deviation_score(dev_df: pd.DataFrame) -> dict:
    active = dev_df[~dev_df["DecisionNorm"].isin(ABSENT)]
    if active.empty:
        return {"pct": 0.0, "deviated": 0, "total": 0}
    return {
        "pct":      round(active["Deviated"].mean() * 100, 1),
        "deviated": int(active["Deviated"].sum()),
        "total":    len(active),
    }


def topic_deviation(mp_pn: int, all_votes: pd.DataFrame,
                    members: pd.DataFrame) -> pd.DataFrame:
    dev_df = mp_deviation(mp_pn, all_votes, members)
    if dev_df.empty:
        return pd.DataFrame()
    dev_df = tag_topics(dev_df)
    rows = []
    for topic, group in dev_df.explode("Topics").groupby("Topics"):
        active = group[~group["DecisionNorm"].isin(ABSENT)]
        if active.empty:
            continue
        rows.append({
            "Thema":              topic,
            "Abweichung (%)":     round(active["Deviated"].mean() * 100, 1),
            "Abweichend":         int(active["Deviated"].sum()),
            "Total":              len(active),
        })
    return pd.DataFrame(rows).sort_values("Abweichung (%)", ascending=False)


# ── Deviation direction: when MP deviated, who did they align with? ──────────

def deviation_alignment(mp_pn: int, all_votes: pd.DataFrame,
                        members: pd.DataFrame) -> pd.DataFrame:
    dev_df = mp_deviation(mp_pn, all_votes, members)
    deviated = dev_df[dev_df["Deviated"] == True]
    if deviated.empty:
        return pd.DataFrame()

    own_frac = members[members["PersonNumber"] == mp_pn].iloc[0]["FractionAbbr"]
    mp_dev_votes = deviated.set_index("GroupKey")["DecisionNorm"]
    dev_keys = set(mp_dev_votes.index)
    active_dec = ["Ja", "Nein", "Enthaltung"]

    results = []
    for frac, fmembers in members[members["FractionAbbr"] != own_frac].groupby("FractionAbbr"):
        frac_pns = set(fmembers["PersonNumber"])
        frac_v = all_votes[
            all_votes["PersonNumber"].isin(frac_pns) &
            all_votes["GroupKey"].isin(dev_keys) &
            all_votes["DecisionNorm"].isin(active_dec)
        ]
        if frac_v.empty:
            continue
        frac_maj = frac_v.groupby("GroupKey")["DecisionNorm"].agg(
            lambda s: s.mode().iloc[0]
        )
        common = mp_dev_votes.index.intersection(frac_maj.index)
        if len(common) == 0:
            continue
        align = int((mp_dev_votes[common] == frac_maj[common]).sum())
        results.append({
            "Fraktion":           frac,
            "Übereinstimmungen":  align,
            "Anteil (%)":         round(align / len(deviated) * 100, 1),
        })
    return pd.DataFrame(results).sort_values("Übereinstimmungen", ascending=False)


# ── Compare two MPs ───────────────────────────────────────────────────────────

def mp_agreement(pn_a: int, pn_b: int, all_votes: pd.DataFrame) -> dict:
    active = ["Ja", "Nein", "Enthaltung"]
    a = all_votes[(all_votes["PersonNumber"] == pn_a) &
                  all_votes["DecisionNorm"].isin(active)].set_index("GroupKey")["DecisionNorm"]
    b = all_votes[(all_votes["PersonNumber"] == pn_b) &
                  all_votes["DecisionNorm"].isin(active)].set_index("GroupKey")["DecisionNorm"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"pct": 0.0, "agree": 0, "total": 0}
    agree = int((a[common] == b[common]).sum())
    return {"pct": round(agree / len(common) * 100, 1), "agree": agree, "total": len(common)}


def diverging_votes(pn_a: int, pn_b: int, all_votes: pd.DataFrame) -> pd.DataFrame:
    active = ["Ja", "Nein", "Enthaltung"]
    a = all_votes[(all_votes["PersonNumber"] == pn_a) &
                  all_votes["DecisionNorm"].isin(active)][
        ["GroupKey", "MeetingDate", "BizShort", "BizNumber", "BizTitle", "DecisionNorm"]
    ].rename(columns={"DecisionNorm": "Vote_A"})
    b = all_votes[(all_votes["PersonNumber"] == pn_b) &
                  all_votes["DecisionNorm"].isin(active)][
        ["GroupKey", "DecisionNorm"]
    ].rename(columns={"DecisionNorm": "Vote_B"})
    merged = a.merge(b, on="GroupKey")
    return merged[merged["Vote_A"] != merged["Vote_B"]].sort_values("MeetingDate", ascending=False)


# ── Fraction-level deviation (all MPs in a fraction) ─────────────────────────

def fraction_all_deviations(fraction_abbr: str, all_votes: pd.DataFrame,
                             members: pd.DataFrame) -> pd.DataFrame:
    """
    Deviation score for every MP in a given fraction.
    Returns DataFrame sorted by deviation descending.
    """
    frac_members = members[members["FractionAbbr"] == fraction_abbr]
    frac_maj = fraction_majority(all_votes, members, fraction_abbr)

    rows = []
    for _, m in frac_members.iterrows():
        pn = m["PersonNumber"]
        mp_votes = all_votes[all_votes["PersonNumber"] == pn].copy()
        if mp_votes.empty:
            continue
        mp_votes["FracMajority"] = mp_votes["GroupKey"].map(frac_maj)
        active = mp_votes[~mp_votes["DecisionNorm"].isin(ABSENT)]
        if active.empty:
            continue
        deviated = (active["DecisionNorm"] != active["FracMajority"]).sum()
        rows.append({
            "PersonNumber":   pn,
            "Name":           f"{m['FirstName']} {m['LastName']}",
            "Kanton":         m["CantonAbbr"],
            "Abweichung (%)": round(deviated / len(active) * 100, 1),
            "Abweichend":     int(deviated),
            "Total":          len(active),
        })
    return pd.DataFrame(rows).sort_values("Abweichung (%)", ascending=False)


# ── Fraction cohesion ─────────────────────────────────────────────────────────

def fraction_cohesion(all_votes: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    rows = []
    active_dec = ["Ja", "Nein", "Enthaltung"]
    for frac, fmembers in members.groupby("FractionAbbr"):
        pns = set(fmembers["PersonNumber"])
        fv = all_votes[
            all_votes["PersonNumber"].isin(pns) &
            all_votes["DecisionNorm"].isin(active_dec)
        ]
        if fv.empty:
            continue
        cohesion = fv.groupby("GroupKey")["DecisionNorm"].nunique().eq(1).mean() * 100
        rows.append({
            "Fraktion":      frac,
            "Kohäsion (%)":  round(float(cohesion), 1),
            "Mitglieder":    len(fmembers),
        })
    return pd.DataFrame(rows).sort_values("Kohäsion (%)", ascending=False)


# ── Topic vote breakdown by fraction ─────────────────────────────────────────

def topic_vote_breakdown(topic: str, all_votes: pd.DataFrame,
                         members: pd.DataFrame) -> pd.DataFrame:
    tv = tag_topics(all_votes)
    tv = tv[tv["Topics"].map(lambda t: topic in t)]
    if tv.empty:
        return pd.DataFrame()
    merged = tv.merge(members[["PersonNumber", "FractionAbbr"]], on="PersonNumber", how="inner")
    if merged.empty:
        return pd.DataFrame()
    counts = (
        merged.groupby(["FractionAbbr", "DecisionNorm"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    return counts


# ── Topic-level helpers ───────────────────────────────────────────────────────

def get_topic_groupkeys(all_votes: pd.DataFrame, topic: str) -> set:
    """GroupKeys for votes whose affair is tagged with the given Curia Vista topic."""
    lookup = _biz_tag_lookup()
    if lookup and "BizNumber" in all_votes.columns:
        matching_bns = {bn for bn, tags in lookup.items() if topic in tags}
        return set(all_votes.loc[all_votes["BizNumber"].isin(matching_bns), "GroupKey"].unique())
    # Fallback: keyword matching
    from topics import TOPICS
    import re as _re
    kws = TOPICS.get(topic, [])
    if not kws:
        return set()
    pat = _re.compile("|".join(_re.escape(k) for k in kws), _re.IGNORECASE)
    text = _topic_text(all_votes)
    return set(all_votes.loc[text.str.contains(pat, na=False), "GroupKey"].unique())


def topic_deviators_all(all_votes: pd.DataFrame, members: pd.DataFrame,
                         positions: pd.DataFrame, topic_gks: set) -> pd.DataFrame:
    """All MPs ranked by deviation rate on a specific topic."""
    tv = all_votes[all_votes["GroupKey"].isin(topic_gks)]
    if tv.empty or positions.empty:
        return pd.DataFrame()
    tp = positions.loc[positions.index.isin(topic_gks)]

    rows = []
    for _, m in members.iterrows():
        pn   = m["PersonNumber"]
        frac = m["FractionAbbr"]
        if pd.isna(frac) or frac not in tp.columns:
            continue
        mv = tv[tv["PersonNumber"] == pn]
        if mv.empty:
            continue
        frac_pos = tp[frac].dropna()
        active   = mv[~mv["DecisionNorm"].isin(ABSENT)].drop_duplicates("GroupKey")
        common   = list(set(active["GroupKey"]) & set(frac_pos.index))
        if len(common) < 3:
            continue
        mp_series = active.set_index("GroupKey").loc[common, "DecisionNorm"]
        dev = int((mp_series != frac_pos[common]).sum())
        rows.append({
            "PersonNumber":   pn,
            "Name":           f"{m['FirstName']} {m['LastName']}",
            "FractionAbbr":   frac,
            "CouncilAbbr":    m["CouncilAbbr"],
            "Kanton":         m["CantonAbbr"],
            "Abweichung (%)": round(dev / len(common) * 100, 1),
            "Abweichend":     dev,
            "Total":          len(common),
        })
    return pd.DataFrame(rows).sort_values("Abweichung (%)", ascending=False)


# ── Geschäft search & detail ──────────────────────────────────────────────────

def search_geschaefte(keyword: str, all_votes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return Geschäfte whose BizShort, BizTitle, or Subject match keyword."""
    av = all_votes if all_votes is not None else load_all_votes()
    if av.empty:
        return pd.DataFrame()
    kw = keyword.lower()
    mask = (
        av["BizShort"].str.lower().str.contains(kw, na=False) |
        av["BizTitle"].str.lower().str.contains(kw, na=False) |
        av["Subject"].str.lower().str.contains(kw, na=False)
    )
    filtered = av[mask & av["BizShort"].str.strip().ne("")]
    if filtered.empty:
        return pd.DataFrame()
    return (
        filtered.groupby("BizShort")
        .agg(
            BizTitle=("BizTitle", "first"),
            Datum=("MeetingDate", "min"),
            Abstimmungen=("GroupKey", "nunique"),
            Stimmende=("PersonNumber", "nunique"),
        )
        .reset_index()
        .sort_values("Datum", ascending=False)
        .head(200)
    )


def geschaeft_votes_enriched(biz_short: str, all_votes: pd.DataFrame,
                              members: pd.DataFrame) -> pd.DataFrame:
    """All vote rows for a BizShort, merged with member info + Abweichler flag."""
    raw = all_votes[all_votes["BizShort"] == biz_short].copy()
    if raw.empty:
        return pd.DataFrame()

    # Compute fraction positions BEFORE merging member columns (avoids FractionAbbr conflict)
    pos = compute_fraction_positions(raw, members)

    # Now enrich with member info
    gv = raw.merge(
        members[["PersonNumber", "FirstName", "LastName",
                 "FractionAbbr", "CouncilAbbr", "CantonAbbr"]],
        on="PersonNumber", how="left",
    )
    gv["Name"] = gv["FirstName"].fillna("") + " " + gv["LastName"].fillna("")

    def _fmaj(row):
        gk, frac = row["GroupKey"], row.get("FractionAbbr")
        if pd.isna(frac) or pos.empty or gk not in pos.index or frac not in pos.columns:
            return None
        v = pos.at[gk, frac]
        return None if pd.isna(v) else v

    gv["FracMajority"] = gv.apply(_fmaj, axis=1)
    active = ~gv["DecisionNorm"].isin(ABSENT)
    gv["Abweichler"] = active & (gv["DecisionNorm"] != gv["FracMajority"])
    return gv


# ── Cross-chamber comparisons ─────────────────────────────────────────────────

def cross_chamber_mp(mp_pn: int, all_votes: pd.DataFrame,
                     members: pd.DataFrame) -> dict:
    """
    Compare MP's voting to same-fraction members in BOTH chambers (by BizShort).
    Returns dict with alignment percentages.
    """
    mp_row = members[members["PersonNumber"] == mp_pn]
    if mp_row.empty:
        return {}
    mi = mp_row.iloc[0]
    frac        = mi["FractionAbbr"]
    own_council = mi["CouncilAbbr"]
    oth_council = "SR" if own_council == "NR" else "NR"

    fm = members[members["FractionAbbr"] == frac]
    own_pns = set(fm.loc[fm["CouncilAbbr"] == own_council,  "PersonNumber"]) - {mp_pn}
    oth_pns = set(fm.loc[fm["CouncilAbbr"] == oth_council, "PersonNumber"])
    if not oth_pns:
        return {}

    av = all_votes[~all_votes["DecisionNorm"].isin(ABSENT)]

    def biz_maj(pns):
        v = av[av["PersonNumber"].isin(pns)]
        return (v.groupby("BizShort")["DecisionNorm"]
                .agg(lambda s: s.mode().iloc[0]) if not v.empty
                else pd.Series(dtype=str))

    mp_pos  = biz_maj({mp_pn})
    own_pos = biz_maj(own_pns)
    oth_pos = biz_maj(oth_pns)

    def agree(a: pd.Series, b: pd.Series):
        c = a.index.intersection(b.index)
        if len(c) == 0:
            return None
        return {"pct": round((a[c] == b[c]).mean() * 100, 1), "total": len(c)}

    return {
        "frac":          frac,
        "own_council":   own_council,
        "oth_council":   oth_council,
        "own_n":         len(own_pns),
        "oth_n":         len(oth_pns),
        "vs_own":        agree(mp_pos, own_pos),
        "vs_oth":        agree(mp_pos, oth_pos),
        "chambers_agree": agree(own_pos, oth_pos),
    }


def cross_chamber_fraction(fraction_abbr: str, all_votes: pd.DataFrame,
                            members: pd.DataFrame) -> pd.DataFrame:
    """NR vs SR majority comparison for a fraction on shared BizShort."""
    fm = members[members["FractionAbbr"] == fraction_abbr]
    nr_pns = set(fm.loc[fm["CouncilAbbr"] == "NR", "PersonNumber"])
    sr_pns = set(fm.loc[fm["CouncilAbbr"] == "SR", "PersonNumber"])
    if not nr_pns or not sr_pns:
        return pd.DataFrame()

    av = all_votes[~all_votes["DecisionNorm"].isin(ABSENT)]
    nr_pos = (av[av["PersonNumber"].isin(nr_pns)]
              .groupby("BizShort")["DecisionNorm"]
              .agg(lambda s: s.mode().iloc[0]).rename("NR"))
    sr_pos = (av[av["PersonNumber"].isin(sr_pns)]
              .groupby("BizShort")["DecisionNorm"]
              .agg(lambda s: s.mode().iloc[0]).rename("SR"))

    df = pd.concat([nr_pos, sr_pos], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    biz_titles = all_votes.groupby("BizShort")["BizTitle"].first()
    biz_dates  = all_votes.groupby("BizShort")["MeetingDate"].min()
    df["BizTitle"] = df.index.map(biz_titles)
    df["Datum"]    = df.index.map(biz_dates)
    df["Einig"]    = df["NR"] == df["SR"]
    return df.reset_index().sort_values("Datum", ascending=False)


# ── Organic coalition analysis ────────────────────────────────────────────────

_FRAC_SCORE_MAP: dict[str, int] = {
    "G": -3, "S": -2, "GL": -1, "M-E": 1, "RL": 2, "V": 3,
}
MAIN_FRACS = list(_FRAC_SCORE_MAP)

# Canonical left-side frozenset → human-readable label
COALITION_PATTERNS: dict[frozenset, str] = {
    # Key = frozenset of the LEFT (lower-score) side fractions.
    # Label: expected winner listed first, then " vs ", then expected loser.
    frozenset({"G", "S"}):                     "GL+M-E+RL+V vs G+S",       # right wins
    frozenset({"G", "S", "GL"}):               "M-E+RL+V vs G+S+GL",       # right wins
    frozenset({"G", "S", "GL", "M-E"}):        "G+S+GL+M-E vs RL+V",       # left-centre wins
    frozenset({"G", "S", "GL", "RL"}):         "G+S+GL+RL vs M-E+V",       # odd alliance wins
    frozenset({"G", "S", "M-E"}):              "G+S+M-E vs GL+RL+V",       # left+M-E wins
    frozenset({"G", "S", "GL", "M-E", "RL"}): "Alle gegen SVP",
}

# Default display order; overridden at runtime by NR count descending.
COALITION_ORDER = [
    "Alle gegen SVP",
    "GL+M-E+RL+V vs G+S",
    "M-E+RL+V vs G+S+GL",
    "Konsens",
    "G+S+GL+M-E vs RL+V",
    "G+S+GL+RL vs M-E+V",
    "G+S+M-E vs GL+RL+V",
]

# Direction a flip pushes the outcome (from the expected-winner's perspective).
COALITION_FLIP_DIRECTION: dict[str, str] = {
    "GL+M-E+RL+V vs G+S":   "nach links",
    "M-E+RL+V vs G+S+GL":   "nach links",
    "G+S+GL+M-E vs RL+V":   "nach rechts",
    "G+S+GL+RL vs M-E+V":   "nach rechts",
    "G+S+M-E vs GL+RL+V":   "nach rechts",
}


def classify_coalition(pos_row: pd.Series) -> tuple:
    """
    Classify a GroupKey into an organic coalition pattern.
    Returns (label, lf, rf, lf_decision, rf_decision).
    lf = frozenset of fractions on the politically-left side (lower FRAC_SCORE avg).
    """
    valid = {f: pos_row[f] for f in MAIN_FRACS
             if f in pos_row.index and pd.notna(pos_row[f]) and pos_row[f] in ("Ja", "Nein")}
    if len(valid) < 3:
        return ("Andere", frozenset(), frozenset(), None, None)

    ja_f  = frozenset(f for f, v in valid.items() if v == "Ja")
    nei_f = frozenset(f for f, v in valid.items() if v == "Nein")
    if not ja_f or not nei_f:
        return ("Konsens", ja_f | nei_f, frozenset(), None, None)

    ja_sc  = sum(_FRAC_SCORE_MAP.get(f, 0) for f in ja_f)  / len(ja_f)
    nei_sc = sum(_FRAC_SCORE_MAP.get(f, 0) for f in nei_f) / len(nei_f)
    lf = ja_f  if ja_sc  <= nei_sc else nei_f
    rf = nei_f if ja_sc  <= nei_sc else ja_f
    lf_dec = "Ja" if lf == ja_f else "Nein"
    rf_dec = "Nein" if lf == ja_f else "Ja"

    # Alle gegen SVP: V alone on either side, 3+ on the other
    if lf == frozenset({"V"}) or rf == frozenset({"V"}):
        if lf == frozenset({"V"}):           # normalise: lf = big coalition
            lf, rf = rf, lf
            lf_dec, rf_dec = rf_dec, lf_dec
        return ("Alle gegen SVP", lf, rf, lf_dec, rf_dec)

    # Konsens: 5+ fracs on one side (after SVP check so Alle-gegen-SVP isn't swallowed)
    if len(lf) >= 5 or len(rf) >= 5:
        return ("Konsens", lf, rf, lf_dec, rf_dec)

    label = COALITION_PATTERNS.get(lf, COALITION_PATTERNS.get(rf, "Andere"))
    return (label, lf, rf, lf_dec, rf_dec)


def compute_coalition_data(all_votes: pd.DataFrame, members: pd.DataFrame,
                            positions: pd.DataFrame) -> pd.DataFrame:
    """
    Per GroupKey: organic coalition type, flip status, chamber, top 2 topics.
    Excludes unclassifiable ('Andere') GroupKeys.
    """
    if positions.empty or all_votes.empty:
        return pd.DataFrame()

    pos_sub = positions[[f for f in MAIN_FRACS if f in positions.columns]]

    # Classify every GroupKey
    classifications: dict[str, tuple] = {}
    for gk in pos_sub.index:
        cls = classify_coalition(pos_sub.loc[gk])
        if cls[0] != "Andere":
            classifications[gk] = cls
    if not classifications:
        return pd.DataFrame()

    active = all_votes[~all_votes["DecisionNorm"].isin(ABSENT)].copy()

    # Actual majority vote (Ja/Nein) per GroupKey
    ja_nein = active[active["DecisionNorm"].isin(("Ja", "Nein"))]
    actual_majority = ja_nein.groupby("GroupKey")["DecisionNorm"].agg(
        lambda s: s.mode().iloc[0] if not s.empty else None
    )
    vote_margin = (
        ja_nein.groupby(["GroupKey", "DecisionNorm"]).size()
        .unstack(fill_value=0)
    )

    # Assign each vote row to a coalition side (lf / rf) via merge
    frac_lookup = members.set_index("PersonNumber")["FractionAbbr"]
    active = active.copy()
    active["FractionAbbr"] = active["PersonNumber"].map(frac_lookup)

    side_rows = [
        {"GroupKey": gk, "FractionAbbr": f, "Side": side}
        for gk, (_, lf, rf, _, _) in classifications.items()
        for f, side in [(f2, "lf") for f2 in lf] + [(f2, "rf") for f2 in rf]
    ]
    side_df = pd.DataFrame(side_rows)

    active_cls = (active[active["GroupKey"].isin(classifications)]
                  .dropna(subset=["FractionAbbr"])
                  .merge(side_df, on=["GroupKey", "FractionAbbr"], how="left"))
    side_counts = (
        active_cls.dropna(subset=["Side"])
        .groupby(["GroupKey", "Side"]).size()
        .unstack(fill_value=0)
        .reindex(columns=["lf", "rf"], fill_value=0)
    )

    # Topic lookup: BizShort → [tag, ...]
    lookup = _biz_tag_lookup()
    biz_topic_map: dict[str, list[str]] = {}
    if lookup and "BizNumber" in all_votes.columns:
        biz_bn = (all_votes[all_votes["BizShort"].notna() & all_votes["BizNumber"].notna()]
                  .drop_duplicates("BizShort")
                  .set_index("BizShort")["BizNumber"])
        for bs, bn in biz_bn.items():
            biz_topic_map[bs] = lookup.get(str(bn or "").strip(), [])

    meta = (all_votes[["GroupKey", "BizShort", "BizTitle", "Subject", "MeetingDate"]]
            .drop_duplicates("GroupKey").set_index("GroupKey"))

    rows = []
    for gk, (label, lf, rf, lf_dec, rf_dec) in classifications.items():
        if gk not in meta.index:
            continue
        m = meta.loc[gk]
        biz_short = str(m.get("BizShort") or "").strip()

        flipped = False
        lf_n = rf_n = 0
        if label != "Konsens" and gk in side_counts.index:
            lf_n = int(side_counts.at[gk, "lf"])
            rf_n = int(side_counts.at[gk, "rf"])
            predicted_dec = lf_dec if lf_n >= rf_n else rf_dec
            actual = actual_majority.get(gk)
            if actual in ("Ja", "Nein"):
                flipped = (predicted_dec != actual)

        margin = 0
        if gk in vote_margin.index:
            j  = int(vote_margin.at[gk, "Ja"])   if "Ja"   in vote_margin.columns else 0
            n  = int(vote_margin.at[gk, "Nein"])  if "Nein" in vote_margin.columns else 0
            margin = abs(j - n)

        topics = biz_topic_map.get(biz_short, [])
        rows.append({
            "GroupKey":      gk,
            "Koalitionstyp": label,
            "lf":            "+".join(sorted(lf, key=lambda f: _FRAC_SCORE_MAP.get(f, 0))),
            "rf":            "+".join(sorted(rf, key=lambda f: _FRAC_SCORE_MAP.get(f, 0))),
            "Flipped":       flipped,
            "Chamber":       "SR" if gk.startswith("SR-") else "NR",
            "lf_count":      lf_n,
            "rf_count":      rf_n,
            "Margin":        margin,
            "BizShort":      biz_short,
            "BizTitle":      str(m.get("BizTitle") or ""),
            "Subject":       str(m.get("Subject") or ""),
            "MeetingDate":   str(m.get("MeetingDate") or ""),
            "Thema1":        topics[0] if topics else "",
            "Thema2":        topics[1] if len(topics) > 1 else "",
        })

    return pd.DataFrame(rows)
