"""
Swiss Parliament Voting Analyser — 7-page Streamlit app.
Run: streamlit run app.py
"""

import os
from urllib.parse import urlencode
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db import init_db, db_stats, DB_PATH, get_conn
from scraper import full_scrape, update_scrape, scrape_sr_votes_direct, backfill_meaning
from analysis import (
    load_members, load_votes_for_mp, load_all_votes, load_votes_for_fraction,
    load_sessions,
    tag_topics, filter_by_topic,
    mp_deviation, deviation_score, topic_deviation,
    compute_fraction_positions, deviation_alignment_fast, fraction_direction_heatmap,
    fraction_majority_strength,
    fraction_cohesion, fraction_all_deviations,
    curia_vista_url, ABSENT,
    get_topic_groupkeys, topic_deviators_all,
    search_geschaefte, geschaeft_votes_enriched,
    cross_chamber_mp, cross_chamber_fraction,
)
from topics import TOPICS
import summarizer

TOPIC_EMOJI = {
    "Aussenpolitik & Entwicklung":  "🌍",
    "Bildung & Forschung":          "📚",
    "Demokratie & Institutionen":   "🏛️",
    "Digitalisierung & Medien":     "💻",
    "Finanzen & Steuern":           "💶",
    "Infrastruktur & Verkehr":      "🏗️",
    "Klima & Energie":              "🌱",
    "Landwirtschaft & Ernährung":   "🌾",
    "Migration & Integration":      "🧭",
    "Nachhaltigkeit & ESG":         "♻️",
    "Sicherheit & Justiz":          "⚖️",
    "Soziales & Gesundheit":        "🏥",
    "Umwelt & Biodiversität":       "🌿",
    "Wirtschaft & Arbeit":          "💼",
}

# ── Fraction label helpers ────────────────────────────────────────────────────

FRAC_NAMES = {
    "RL":  "FDP",
    "S":   "SP",
    "V":   "SVP",
    "G":   "Grüne",
    "GL":  "GLP",
    "M-E": "Mitte-EVP",
    "BD":  "BDP",
}

# Left-to-right political spectrum order
FRAC_ORDER = ["G", "S", "GL", "M-E", "RL", "V"]

def frac_order_labels() -> list:
    """Fraction labels in left-to-right order (unknowns appended at end)."""
    known = [frac_label(a) for a in FRAC_ORDER]
    return known  # Plotly silently skips labels not present in the data

# SRF party color scheme — https://github.com/srfdata/swiss-party-colors
FRAC_COLORS = {
    "RL":  "#3872B5",  # FDP
    "S":   "#F0554D",  # SP
    "V":   "#4B8A3E",  # SVP
    "G":   "#84B547",  # Grüne
    "GL":  "#C4C43D",  # GLP
    "M-E": "#D6862B",  # Mitte-EVP
    "BD":  "#B8B8B8",  # BDP (aufgelöst)
}

def frac_label(abbr: str) -> str:
    return FRAC_NAMES.get(str(abbr), str(abbr))

def frac_color_map(*extra_dicts) -> dict:
    """Color map keyed by both abbreviation and readable label, for use in plotly charts."""
    m = {}
    for abbr, color in FRAC_COLORS.items():
        m[abbr]             = color
        m[frac_label(abbr)] = color
    for d in extra_dicts:
        m.update(d)
    return m

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wer stimmt wie?",
    page_icon="🏛️",
    layout="wide",
)

st.markdown("""
<style>
/* ── Sidebar nav buttons ── */
div[data-testid="stSidebar"] div[data-testid="stButton"] button {
    display: block !important;
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    color: inherit !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 0.9rem !important;
    text-align: left !important;
    border-radius: 0.4rem !important;
    width: 100% !important;
    transition: background 0.12s !important;
    cursor: pointer !important;
}
div[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: inherit !important;
}
div[data-testid="stSidebar"] div[data-testid="stButton"] button:focus {
    box-shadow: none !important;
    outline: none !important;
}
div[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin-bottom: 0 !important;
}
/* ── Plotly charts: default arrow cursor instead of move/crosshair ── */
.js-plotly-plot .plotly .cursor-move,
.js-plotly-plot .plotly .cursor-crosshair,
.js-plotly-plot .plotly .cursor-ew-resize,
.js-plotly-plot .plotly .cursor-ns-resize {
    cursor: default !important;
}
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def get_members():
    return load_members()

@st.cache_data(ttl=1800, show_spinner=False)
def get_all_votes():
    return load_all_votes()

@st.cache_data(ttl=1800, show_spinner=False)
def get_mp_votes(pn: int):
    av = get_all_votes()
    return av[av["PersonNumber"] == pn].copy()

@st.cache_data(ttl=1800, show_spinner=False)
def get_fraction_votes(abbr: str):
    av = get_all_votes()
    m  = get_members()
    pns = set(m.loc[m["FractionAbbr"] == abbr, "PersonNumber"])
    return av[av["PersonNumber"].isin(pns)].copy()

@st.cache_data(ttl=3600, show_spinner=False)
def get_sessions():
    return load_sessions()

@st.cache_data(ttl=3600, show_spinner="Berechne Fraktionspositionen…")
def get_positions():
    return compute_fraction_positions(get_all_votes(), get_members())

@st.cache_data(ttl=3600, show_spinner="Berechne Abweichungsraten…")
def get_all_mp_deviations():
    """Deviation rate for every MP across all votes, using positions matrix for speed."""
    av  = get_all_votes()
    m   = get_members()
    pos = get_positions()

    active_dec = {"Ja", "Nein", "Enthaltung"}
    active = av[av["DecisionNorm"].isin(active_dec)].copy()

    mp_fracs = m.set_index("PersonNumber")["FractionAbbr"]
    active["FractionAbbr"] = active["PersonNumber"].map(mp_fracs)

    pos_long = pos.stack().reset_index()
    pos_long.columns = ["GroupKey", "FractionAbbr", "FracMajority"]

    active = active.merge(pos_long, on=["GroupKey", "FractionAbbr"], how="left")
    active = active.dropna(subset=["FracMajority"])
    active["Deviated"] = active["DecisionNorm"] != active["FracMajority"]

    agg = active.groupby("PersonNumber").agg(
        Total=("DecisionNorm", "count"),
        Deviated=("Deviated", "sum"),
    ).reset_index()
    agg["Abweichung (%)"] = (agg["Deviated"] / agg["Total"] * 100).round(1)

    info = m[["PersonNumber", "FirstName", "LastName", "FractionAbbr",
              "CouncilAbbr", "CantonAbbr"]]
    agg = agg.merge(info, on="PersonNumber", how="left")
    agg["Name"] = agg["FirstName"] + " " + agg["LastName"]
    return agg

@st.cache_data(ttl=3600, show_spinner="Berechne Themen-Kohäsionsmatrix…")
def get_topic_fraction_cohesion():
    """Returns a (n_topics × n_fracs) DataFrame of cohesion % for the heatmap."""
    av = get_all_votes()
    m  = get_members()
    active_dec = ["Ja", "Nein", "Enthaltung"]
    rows = []
    for topic_name in TOPICS.keys():
        gks = get_topic_groupkeys(av, topic_name)
        t_votes = av[av["GroupKey"].isin(gks)]
        for frac in FRAC_ORDER:
            frac_pns = set(m.loc[m["FractionAbbr"] == frac, "PersonNumber"])
            fv = t_votes[
                t_votes["PersonNumber"].isin(frac_pns) &
                t_votes["DecisionNorm"].isin(active_dec)
            ]
            if fv.empty:
                coh = float("nan")
            else:
                coh = round(float(
                    fv.groupby("GroupKey")["DecisionNorm"].nunique().eq(1).mean() * 100
                ), 1)
            rows.append({"Thema": topic_name, "Fraktion": frac_label(frac), "Kohäsion (%)": coh})
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def get_mp_fraction_alignment():
    """246×6 matrix: % of each MP's active votes agreeing with each fraction's majority."""
    av  = get_all_votes()
    m   = get_members()
    pos = get_positions()

    active_dec = {"Ja", "Nein", "Enthaltung"}
    active = av[av["DecisionNorm"].isin(active_dec)][
        ["PersonNumber", "GroupKey", "DecisionNorm"]
    ]
    pos_long = pos[FRAC_ORDER].stack().reset_index()
    pos_long.columns = ["GroupKey", "FracCol", "FracMajority"]

    merged = active.merge(pos_long, on="GroupKey", how="inner")
    merged["Agrees"] = merged["DecisionNorm"] == merged["FracMajority"]

    agg = (
        merged.groupby(["PersonNumber", "FracCol"])["Agrees"]
        .mean()
        .mul(100).round(1)
        .reset_index()
        .rename(columns={"Agrees": "Agreement (%)"})
    )
    pivot = agg.pivot(index="PersonNumber", columns="FracCol", values="Agreement (%)").reset_index()
    pivot.columns.name = None

    info = m[["PersonNumber", "FirstName", "LastName", "FractionAbbr", "CouncilAbbr", "CantonAbbr"]]
    pivot = pivot.merge(info, on="PersonNumber", how="left")
    pivot["Name"] = pivot["FirstName"] + " " + pivot["LastName"]
    return pivot

@st.cache_data(ttl=3600, show_spinner=False)
def get_mp_pca():
    """PCA on the 6D fraction-alignment vectors → 2D political compass."""
    import numpy as np
    aln = get_mp_fraction_alignment()
    if aln.empty:
        return pd.DataFrame()

    frac_cols = [f for f in FRAC_ORDER if f in aln.columns]
    # Drop members with no valid L52 vote data (all fraction columns NaN)
    aln = aln[~aln[frac_cols].isna().all(axis=1)].copy()
    if aln.empty:
        return pd.DataFrame()

    mat = aln[frac_cols].values.astype(float)

    col_means = np.nanmean(mat, axis=0)
    for j in range(mat.shape[1]):
        mat[np.isnan(mat[:, j]), j] = col_means[j]

    mat_c = mat - mat.mean(axis=0)
    U, S, _ = np.linalg.svd(mat_c, full_matrices=False)
    coords = U[:, :2] * S[:2]

    result = pd.DataFrame({
        "PersonNumber": aln["PersonNumber"].values,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })
    info = get_members()[["PersonNumber", "FirstName", "LastName",
                           "FractionAbbr", "CouncilAbbr", "CantonAbbr"]]
    result = result.merge(info, on="PersonNumber", how="left")
    result["Name"] = result["FirstName"] + " " + result["LastName"]
    dev = get_all_mp_deviations()[["PersonNumber", "Abweichung (%)"]]
    result = result.merge(dev, on="PersonNumber", how="left")
    return result

@st.cache_data(ttl=3600, show_spinner=False)
def get_vote_polarity_scatter():
    """Per roll-call: % of non-left members voting with left, and non-right with right.
    Center (GL, M-E) is included in both axes."""
    import numpy as np
    av  = get_all_votes()
    m   = get_members()
    pos = get_positions()
    if av.empty or pos.empty:
        return pd.DataFrame()

    left_fracs  = {"G", "S"}
    right_fracs = {"V", "RL"}
    active_dec  = {"Ja", "Nein", "Enthaltung"}

    def _bloc_pos(fracs):
        cols = [f for f in fracs if f in pos.columns]
        if not cols:
            return pd.Series(dtype=str)
        return pos[cols].apply(
            lambda row: row.dropna().mode().iloc[0] if row.notna().any() else None, axis=1
        )

    lpos = _bloc_pos(left_fracs)
    rpos = _bloc_pos(right_fracs)
    polar_gks = set(pos.index[lpos.notna() & rpos.notna() & (lpos != rpos)])

    active = av[av["DecisionNorm"].isin(active_dec) & av["GroupKey"].isin(polar_gks)].copy()
    active = active.merge(m[["PersonNumber", "FractionAbbr"]], on="PersonNumber", how="inner")
    active["LeftPos"]  = active["GroupKey"].map(lpos)
    active["RightPos"] = active["GroupKey"].map(rpos)

    # X-axis: center + right-bloc members who voted with the left majority
    not_left  = {"V", "RL", "GL", "M-E"}
    # Y-axis: left-bloc + center members who voted with the right majority
    not_right = {"G", "S", "GL", "M-E"}

    active["DevLeft"]    = (active["FractionAbbr"].isin(not_left) &
                            (active["DecisionNorm"] == active["LeftPos"])).astype(int)
    active["IsNotLeft"]  = active["FractionAbbr"].isin(not_left).astype(int)
    active["DevRight"]   = (active["FractionAbbr"].isin(not_right) &
                            (active["DecisionNorm"] == active["RightPos"])).astype(int)
    active["IsNotRight"] = active["FractionAbbr"].isin(not_right).astype(int)

    agg = active.groupby("GroupKey").agg(
        DevLeft=("DevLeft", "sum"),
        TotalNotLeft=("IsNotLeft", "sum"),
        DevRight=("DevRight", "sum"),
        TotalNotRight=("IsNotRight", "sum"),
    ).reset_index()

    agg["PctDevLeft"]  = (agg["DevLeft"]  / agg["TotalNotLeft"].clip(lower=1)  * 100).round(1)
    agg["PctDevRight"] = (agg["DevRight"] / agg["TotalNotRight"].clip(lower=1) * 100).round(1)

    meta = (av[["GroupKey", "BizShort", "BizTitle", "Subject", "MeetingDate"]]
            .drop_duplicates("GroupKey"))
    return agg.merge(meta, on="GroupKey", how="left")

@st.cache_data(ttl=3600, show_spinner=False)
def get_direction_magnitude_scatter():
    """Per roll-call: Y = overall deviation rate, X = net political gravity of deviators,
    color = coalition type. Only polarising votes (left ≠ right bloc)."""
    import numpy as np
    av  = get_all_votes()
    m   = get_members()
    pos = get_positions()
    if av.empty or pos.empty:
        return pd.DataFrame()

    FRAC_SCORE  = {"G": -3, "S": -2, "GL": -1, "M-E": 1, "RL": 2, "V": 3}
    left_fracs  = {"G", "S"}
    right_fracs = {"V", "RL"}
    active_dec  = {"Ja", "Nein", "Enthaltung"}

    def _bloc_pos(fracs):
        cols = [f for f in fracs if f in pos.columns]
        if not cols:
            return pd.Series(dtype=str)
        return pos[cols].apply(
            lambda row: row.dropna().mode().iloc[0] if row.notna().any() else None, axis=1
        )

    lpos = _bloc_pos(left_fracs)
    rpos = _bloc_pos(right_fracs)
    polar_gks = set(pos.index[lpos.notna() & rpos.notna() & (lpos != rpos)])

    active = av[av["DecisionNorm"].isin(active_dec) & av["GroupKey"].isin(polar_gks)].copy()
    active = active.merge(m[["PersonNumber", "FractionAbbr"]], on="PersonNumber", how="inner")

    frac_cols = [f for f in FRAC_SCORE if f in pos.columns]
    pos_long = pos[frac_cols].stack().reset_index()
    pos_long.columns = ["GroupKey", "FractionAbbr", "FracMajority"]
    active = active.merge(pos_long, on=["GroupKey", "FractionAbbr"], how="left")
    active = active.dropna(subset=["FracMajority"])
    active["Deviated"] = active["DecisionNorm"] != active["FracMajority"]

    active["LeftVote"]     = active["GroupKey"].map(lpos)
    active["RightVote"]    = active["GroupKey"].map(rpos)
    active["FracScoreAbs"] = active["FractionAbbr"].map(lambda f: abs(FRAC_SCORE.get(f, 0)))

    # For each deviating MP: -|score| if they crossed to the left side, +|score| to the right
    left_dev  = active["Deviated"] & (active["DecisionNorm"] == active["LeftVote"])
    right_dev = active["Deviated"] & (active["DecisionNorm"] == active["RightVote"])
    active["DirectionContrib"] = float("nan")
    active.loc[left_dev,  "DirectionContrib"] = -active.loc[left_dev,  "FracScoreAbs"]
    active.loc[right_dev, "DirectionContrib"] =  active.loc[right_dev, "FracScoreAbs"]

    dev_agg = (
        active.groupby("GroupKey")
        .agg(NActive=("Deviated", "count"), NDeviated=("Deviated", "sum"))
        .reset_index()
    )
    dev_agg["DeviationPct"] = (
        dev_agg["NDeviated"] / dev_agg["NActive"].clip(lower=1) * 100
    ).round(1)

    grav_agg = (
        active[active["Deviated"] & active["DirectionContrib"].notna()]
        .groupby("GroupKey")["DirectionContrib"]
        .mean().div(3).round(3).reset_index()
        .rename(columns={"DirectionContrib": "NetGravity"})
    )

    agg = dev_agg.merge(grav_agg, on="GroupKey", how="left")
    agg["NetGravity"] = agg["NetGravity"].fillna(0)

    pos_sub = pos[frac_cols].copy()

    def _classify(gk):
        if gk not in pos_sub.index:
            return "Andere"
        row = pos_sub.loc[gk]
        valid = {f: row[f] for f in FRAC_SCORE
                 if f in row.index and pd.notna(row[f]) and row[f] in ("Ja", "Nein")}
        if len(valid) < 3:
            return "Andere"
        ja_f  = {f for f, v in valid.items() if v == "Ja"}
        nei_f = {f for f, v in valid.items() if v == "Nein"}
        if not ja_f or not nei_f:
            return "Konsens"
        ja_sc  = sum(FRAC_SCORE[f] for f in ja_f)  / len(ja_f)
        nei_sc = sum(FRAC_SCORE[f] for f in nei_f) / len(nei_f)
        lf = ja_f  if ja_sc  <= nei_sc else nei_f
        rf = nei_f if ja_sc  <= nei_sc else ja_f
        if len(lf) >= 5 or len(rf) >= 5:
            return "Konsens"
        for side in [lf, rf]:
            if (side & {"G", "S"}) and (side & {"V", "RL"}):
                return "Unheilige Allianz"
        ctr   = {"GL", "M-E"}
        cl    = ctr & lf
        cr    = ctr & rf
        has_l = bool(lf & {"G", "S"})
        has_r = bool(rf & {"RL", "V"})
        if has_l and has_r:
            if cl and not cr:
                return "Mitte kippt links"
            if cr and not cl:
                return "Mitte kippt rechts"
            return "Parteilinien"
        return "Andere"

    agg["Koalitionstyp"] = agg["GroupKey"].apply(_classify)

    meta = (av[["GroupKey", "BizShort", "BizTitle", "Subject", "MeetingDate"]]
            .drop_duplicates("GroupKey"))
    return agg.merge(meta, on="GroupKey", how="left")

@st.cache_data(ttl=3600, show_spinner=False)
def get_sunburst_data():
    """Build ids/labels/parents/values/colors arrays for the Übersicht sunburst."""
    dev = get_all_mp_deviations()
    if dev.empty:
        return [], [], [], [], []
    ids, labels, parents, values, colors = [], [], [], [], []
    for frac in FRAC_ORDER:
        fdata = dev[dev["FractionAbbr"] == frac]
        if fdata.empty:
            continue
        fclr  = FRAC_COLORS.get(frac, "#888888")
        flbl  = frac_label(frac)
        ids.append(frac);       labels.append(flbl)
        parents.append("");     values.append(int(fdata["Total"].sum()))
        colors.append(fclr)
        for _, mp in fdata.iterrows():
            mp_id = f"{frac}_{int(mp['PersonNumber'])}"
            inl   = max(0, int(mp["Total"]) - int(mp["Deviated"]))
            dv    = int(mp["Deviated"])
            ids.append(mp_id);        labels.append(mp["Name"])
            parents.append(frac);     values.append(int(mp["Total"]))
            colors.append(fclr)
            ids.append(mp_id+"_i");   labels.append("Fraktionslinie")
            parents.append(mp_id);    values.append(inl)
            colors.append(fclr)
            ids.append(mp_id+"_d");   labels.append("Abweichend")
            parents.append(mp_id);    values.append(dv)
            colors.append("#cccccc")
    return ids, labels, parents, values, colors

def clear_cache():
    for fn in (get_members, get_all_votes, get_mp_votes,
               get_fraction_votes, get_sessions, get_positions,
               get_all_mp_deviations, get_topic_fraction_cohesion,
               get_mp_fraction_alignment, get_mp_pca, get_sunburst_data,
               get_vote_polarity_scatter, get_direction_magnitude_scatter):
        fn.clear()

def _has_db() -> bool:
    from db import DB_PATH
    return DB_PATH.exists()

def _migrate_decisions():
    if not _has_db():
        return
    with get_conn() as conn:
        conn.execute("UPDATE votes SET DecisionNorm='Entschuldigt' WHERE DecisionNorm='6'")
        conn.execute("UPDATE votes SET DecisionNorm='Nicht teilgenommen' WHERE DecisionNorm='7'")

# ── UI helpers ────────────────────────────────────────────────────────────────

COLORS = {
    "Ja": "#2ecc71", "Nein": "#e74c3c", "Enthaltung": "#f39c12",
    "Abwesend": "#95a5a6", "Entschuldigt": "#bdc3c7",
    "Nicht teilgenommen": "#d5d8dc", "Präsident": "#8e44ad", "—": "#cccccc",
}

def member_label(row):
    return f"{row['FirstName']} {row['LastName']} ({frac_label(row['FractionAbbr'])}, {row['CantonAbbr']})"

def mp_selector(label, key, members_df, target_pn=None):
    labels = [member_label(r) for _, r in members_df.iterrows()]
    pns    = [int(r["PersonNumber"]) for _, r in members_df.iterrows()]
    idx = 0
    if target_pn and int(target_pn) in pns:
        idx = pns.index(int(target_pn))
    choice = st.selectbox(label, labels, index=idx, key=key)
    return pns[labels.index(choice)]

def navigate_to_mp(pn: int):
    st.query_params["page"] = "Mitglied"
    st.query_params["mp"]   = str(pn)
    if "q" in st.query_params:
        del st.query_params["q"]
    st.session_state["nav_radio"] = "Mitglied"
    st.session_state["_nav_pn"]   = int(pn)
    st.rerun()

def navigate_to_geschaeft(biz_short: str):
    st.query_params["page"] = "Geschäft"
    st.query_params["q"]    = str(biz_short)
    if "mp" in st.query_params:
        del st.query_params["mp"]
    st.session_state["_geschaeft_goto"] = str(biz_short)
    st.session_state["nav_radio"]       = "Geschäft"
    st.rerun()

def _mp_click_nav(ev, key_suffix, members_df):
    """Show a one-line navigation widget when a chart point (with PersonNumber
    in customdata[0]) is clicked. Returns True when nav was triggered."""
    if not (ev and ev.selection and ev.selection.points):
        return False
    try:
        pn = int(ev.selection.points[0]["customdata"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    row = members_df[members_df["PersonNumber"] == pn]
    if row.empty:
        return False
    r = row.iloc[0]
    _c1, _c2 = st.columns([5, 1])
    _c1.info(
        f"**{r['FirstName']} {r['LastName']}** · "
        f"{frac_label(r['FractionAbbr'])} · {r['CouncilAbbr']} · {r['CantonAbbr']}"
    )
    _href_mp = "?" + urlencode({"page": "Mitglied", "mp": str(pn)})
    _c2.markdown(
        f'<a href="{_href_mp}" target="_self" style="display:inline-block;padding:0.35rem 0.7rem;'
        f'background:#f63366;color:white;border-radius:0.3rem;text-decoration:none;'
        f'font-size:0.875rem;font-weight:500;">→ Mitglied</a>',
        unsafe_allow_html=True,
    )
    return True

def _biz_click_nav(ev, key_suffix):
    """Show a one-line navigation widget when a chart point (with BizShort
    in customdata[2]) is clicked."""
    if not (ev and ev.selection and ev.selection.points):
        return False
    try:
        pt = ev.selection.points[0]
        biz_short = pt["customdata"][2]
        title     = pt["customdata"][0] or pt["customdata"][1] or ""
    except (KeyError, IndexError, TypeError):
        return False
    _c1, _c2 = st.columns([5, 1])
    _c1.info(f"Geschäft **{biz_short}**" + (f" — {str(title)[:80]}" if title else ""))
    _href_biz = "?" + urlencode({"page": "Geschäft", "q": str(biz_short)})
    _c2.markdown(
        f'<a href="{_href_biz}" target="_self" style="display:inline-block;padding:0.35rem 0.7rem;'
        f'background:#f63366;color:white;border-radius:0.3rem;text-decoration:none;'
        f'font-size:0.875rem;font-weight:500;">→ Geschäft</a>',
        unsafe_allow_html=True,
    )
    return True

def ai_button(key, fn, *args):
    if st.button("🤖 KI-Zusammenfassung", key=key):
        with st.spinner("Claude analysiert…"):
            try:
                st.info(fn(*args))
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Fehler: {e}")


def check_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        inp = st.sidebar.text_input("ANTHROPIC_API_KEY", type="password",
                                    help="Lokal gespeichert in .env")
        if inp:
            os.environ["ANTHROPIC_API_KEY"] = inp
            (DB_PATH.parent / ".env").write_text(f"ANTHROPIC_API_KEY={inp}\n", encoding="utf-8")
            st.sidebar.success("Key gespeichert")

def no_data_stop():
    st.warning("Keine Daten — bitte zuerst scrapen (Seite 'Daten').")
    st.stop()

def _make_curia_url(biz_short, biz_number=""):
    url = curia_vista_url(str(biz_short or ""), str(biz_number or ""))
    if not url:
        bs = str(biz_short or "").strip()
        if bs.isdigit() and 6 <= len(bs) <= 8:
            url = ("https://www.parlament.ch/de/ratsbetrieb/"
                   "suche-curia-vista/geschaeft?AffairId=" + bs.zfill(8))
    return url

def vote_table(df, extra_cols=None, height=400):
    df = df.copy()
    df["🔗"] = df.apply(
        lambda r: _make_curia_url(r.get("BizShort", ""), r.get("BizNumber", "")), axis=1
    )
    base = ["MeetingDate", "BizShort", "Subject", "BizTitle", "DecisionNorm"]
    if extra_cols:
        base += extra_cols
    cols = [c for c in base + ["🔗"] if c in df.columns]
    renamed = {
        "MeetingDate": "Datum", "BizShort": "Geschäft",
        "Subject": "Frage", "BizTitle": "Titel", "DecisionNorm": "Entscheid",
    }
    display = df[cols].rename(columns=renamed)
    if "Datum" in display.columns:
        display = display.sort_values("Datum", ascending=False)
    cfg = {"🔗": st.column_config.LinkColumn("Curia Vista", display_text="↗")}
    st.dataframe(display, column_config=cfg, use_container_width=True, height=height)

# ── Methodology ───────────────────────────────────────────────────────────────

_METHODOLOGY = """
**Was zählt als Abweichung?**
Eine Abstimmung gilt als Abweichung, wenn das Mitglied anders votierte als die
*Mehrheit der anwesenden* Fraktionsmitglieder auf derselben Abstimmung.

**Was wird ausgeschlossen?**
Abwesend, Entschuldigt und Nicht-teilgenommen werden *nicht* als Abweichung gewertet —
sie sind kein aktives Votum gegen die Fraktion. Nur Ja, Nein und Enthaltung fliessen
in die Abweichungsberechnung ein.

**Mehrere Abstimmungen pro Geschäft:** Ein Geschäft (z.B. «23.456») kann mehrere
Abstimmungen umfassen — etwa Eintreten, Rückweisung, Detailberatungsartikel und
Gesamtabstimmung. Jede dieser Abstimmungen wird separat gezählt und ausgewertet.

**Fraktionsmehrheit:** «Ja (87%)» zeigt, wie klar die Mehrheit innerhalb der Fraktion war.
Bei knapper Mehrheit (z.B. 52%) ist eine abweichende Stimme weniger auffällig als bei 95%.

**Datenquelle:** Swiss Parliament OData API (parlament.ch), Legislatur 52 (ab Dez. 2023).
"""

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🏛️ Wer stimmt wie?")
check_api_key()

_pages = ["Home", "Fraktion", "Mitglied", "Thema", "Geschäft"]

# URL is source of truth for page and detail state (survives browser reload/back)
_page_from_url = st.query_params.get("page", "")
if _page_from_url in _pages:
    st.session_state["nav_radio"] = _page_from_url
    _mp_param = st.query_params.get("mp")
    if _mp_param:
        try:
            st.session_state["_nav_pn"] = int(_mp_param)
        except (ValueError, TypeError):
            pass
    _q_param = st.query_params.get("q")
    if _q_param:
        st.session_state["_geschaeft_goto"] = _q_param

if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "Home"
page = st.session_state["nav_radio"]

# All nav items are <a> links so the browser adds history entries.
# Clicking the active item navigates to ?page=X (no detail params) → resets to landing.
_nav_html = ["<div style='margin-top:6px'>"]
for _p in _pages:
    _href = "?" + urlencode({"page": _p})
    if _p == page:
        _nav_html.append(
            f'<a href="{_href}" target="_self" style="display:block;padding:0.45rem 0.75rem 0.45rem 0.65rem;'
            f'background:rgba(255,255,255,0.18);font-weight:700;'
            f'border-left:3px solid #f63366;border-radius:0.4rem;'
            f'margin:0 0 2px 0;font-size:1rem;line-height:1.6;'
            f'color:inherit;text-decoration:none;">{_p}</a>'
        )
    else:
        _nav_html.append(
            f'<a href="{_href}" target="_self" style="display:block;padding:0.45rem 0.9rem;'
            f'border-radius:0.4rem;margin:0 0 2px 0;font-size:1rem;line-height:1.6;'
            f'color:inherit;text-decoration:none;" '
            f'onmouseover="this.style.background=\'rgba(255,255,255,0.08)\'" '
            f'onmouseout="this.style.background=\'none\'">{_p}</a>'
        )
_nav_html.append("</div>")
st.sidebar.markdown("".join(_nav_html), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 1. HOME
# ════════════════════════════════════════════════════════════════════════════

if page == "Home":
    st.title("Wer stimmt wie?")
    st.markdown("##### Abstimmungsanalyse Schweizer Parlamentarier:innen")

    st.markdown(
        "Diese App analysiert das Abstimmungsverhalten aller Mitglieder des Schweizer Parlaments "
        "(National- und Ständerat, Legislatur 52, ab Dezember 2023). "
        "Im Zentrum steht die Frage: Wer weicht wie oft von der Fraktionslinie ab — "
        "bei welchen Themen, in welche Richtung, und im Vergleich zu anderen Fraktionen?"
    )

    # ── PCA political compass ─────────────────────────────────────────────
    with st.spinner("Berechne politische Landkarte…"):
        _pca = get_mp_pca()

    if not _pca.empty:
        _pca_plot = _pca[_pca["FractionAbbr"].isin(FRAC_ORDER)].copy()
        _pca_plot["Fraktion"] = _pca_plot["FractionAbbr"].apply(frac_label)
        _fig_pca = px.scatter(
            _pca_plot,
            x="PC1", y="Abweichung (%)",
            color="Fraktion",
            color_discrete_map=frac_color_map(),
            hover_name="Name",
            hover_data={
                "FractionAbbr": False,
                "Fraktion": True,
                "CouncilAbbr": True,
                "CantonAbbr": True,
                "Abweichung (%)": True,
                "PC1": False,
            },
            category_orders={"Fraktion": frac_order_labels()},
            custom_data=["PersonNumber"],
            labels={"CouncilAbbr": "Rat", "CantonAbbr": "Kanton",
                    "Abweichung (%)": "Abweichung vom Fraktionskurs (%)"},
        )
        _fig_pca.update_traces(
            marker=dict(size=9, opacity=0.82, line=dict(width=0.5, color="white"))
        )
        _fig_pca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend_title="",
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            xaxis=dict(showticklabels=False, showgrid=True, zeroline=False,
                       title="← Links  ·  Politische Achse  ·  Rechts →"),
            yaxis=dict(showgrid=True, zeroline=False,
                       title="Abweichung vom Fraktionskurs (%)"),
            height=480,
            margin=dict(l=50, r=20, t=20, b=60),
        )
        _ev_pca = st.plotly_chart(
            _fig_pca, use_container_width=True,
            key="home_pca", on_select="rerun",
        )
        st.caption(
            "Jeder Punkt ist ein Parlamentsmitglied. "
            "X-Achse (horizontale Position): politische Ausrichtung, berechnet aus dem tatsächlichen "
            "Abstimmungsverhalten gegenüber allen sechs Fraktionen — also wie oft jemand mit den Grünen, "
            "der SP, der GLP, der Mitte, der FDP und der SVP übereinstimmt. "
            "Diese sechs Prozentwerte (dieselben wie in der Heatmap unter «Mitglied») werden per "
            "Hauptkomponentenanalyse auf eine Achse verdichtet: Links = höhere Übereinstimmung mit "
            "linken Fraktionen, rechts = höhere Übereinstimmung mit rechten. "
            "Ein SVP-Mitglied, das weiter links erscheint als andere SVP-Mitglieder, stimmt "
            "messbar öfter mit linken Fraktionen überein als seine Fraktionskolleg:innen. "
            "Y-Achse (vertikale Position): Abweichungsrate vom Fraktionskurs — wie oft jemand "
            "anders stimmt als die Mehrheit der eigenen Fraktion. "
            "Punkt anklicken für Direktnavigation."
        )
        _mp_click_nav(_ev_pca, "home_pca", get_members())


    st.markdown("---")
    st.subheader("So verwendest du die App")
    st.markdown(
        "**Fraktion** — Übersicht über alle Fraktionen: Kohäsionsvergleich, Abweichungsraten "
        "der einzelnen Mitglieder, Richtungsanalyse (mit welchen anderen Fraktionen wird bei "
        "Abweichungen gestimmt?) und ein Kammernvergleich NR vs. SR.\n\n"
        "**Mitglied** — Detailprofil eines Parlamentsmitglieds: Gesamtabweichung vom "
        "Fraktionskurs, Abweichung nach Thema und Session, Richtungsanalyse, vollständige "
        "Abstimmungshistorie — und ein Vergleich mit den Fraktionskolleg:innen in der "
        "anderen Kammer.\n\n"
        "**Thema** — Wähle ein politisches Thema (z.B. Klima, Migration, Soziales) und sieh, "
        "wie geschlossen die Fraktionen abstimmten und wer am stärksten von der eigenen Linie abwich.\n\n"
        "**Geschäft** — Suche nach einer Geschäftsnummer (z.B. `23.456`) oder einem "
        "Stichwort und sieh, wie jede Fraktion und jedes Mitglied bei jeder einzelnen "
        "Abstimmung votierte."
    )

    st.warning(
        "**🤖 KI-Zusammenfassungen** — Jede Seite bietet eine KI-Zusammenfassung via Claude. "
        "Bitte diese Funktion sparsam benutzen: die Kosten werden von Sam's persönlichem API-Budget gedeckt."
    )

    st.markdown("---")
    st.subheader("Methodik")
    st.info(_METHODOLOGY)

    st.markdown("---")
    st.subheader("Hintergrund & Feedback")
    st.markdown(
        "Diese App ist ein persönliches Nebenprojekt — kein professionelles Produkt, "
        "keine Organisation dahinter. Ziel ist es, etwas mehr Transparenz über das tatsächliche "
        "Abstimmungsverhalten im Parlament zu schaffen, jenseits von Parteiprogrammen und "
        "Selbstdarstellung.\n\n"
        "Die Infrastruktur- und API-Kosten werden aus dem persönlichen Budget gedeckt. "
        "Bitte teile die App nur mit Personen, denen du vertraust, und geh sparsam mit den "
        "KI-Zusammenfassungen um — jede Anfrage verursacht Kosten."
    )

    import streamlit.components.v1 as _components
    _components.html("""
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: "Source Sans Pro", sans-serif; }
  body { background: transparent; }
  input, textarea {
    display: block; width: 100%; padding: 8px 12px; margin-bottom: 10px;
    border: 1px solid rgba(150,150,150,0.35); border-radius: 6px;
    background: rgba(150,150,150,0.07); color: inherit;
    font-size: 14px; font-family: inherit; outline: none; resize: vertical;
  }
  input:focus, textarea:focus { border-color: #f63366; }
  ::placeholder { color: rgba(150,150,150,0.7); }
  button {
    background: #f63366; color: #fff; border: none; border-radius: 6px;
    padding: 8px 22px; font-size: 14px; font-family: inherit; cursor: pointer;
  }
  button:hover { background: #d42d58; }
  #ok { display:none; color: #2ecc71; font-size: 14px; padding-top: 6px; }
  #err { display:none; color: #e74c3c; font-size: 13px; margin-bottom: 6px; }
</style>
<form id="fb">
  <input type="text" id="nm" placeholder="Name (optional)">
  <textarea id="msg" rows="4" placeholder="Nachricht"></textarea>
  <div id="err">Bitte schreib eine Nachricht.</div>
  <button type="submit">Absenden</button>
</form>
<div id="ok">Danke! Dein E-Mail-Programm öffnet sich gleich mit dem Text.</div>
<script>
document.getElementById('fb').addEventListener('submit', function(e) {
  e.preventDefault();
  var msg = document.getElementById('msg').value.trim();
  if (!msg) { document.getElementById('err').style.display='block'; return; }
  document.getElementById('err').style.display = 'none';
  var name = document.getElementById('nm').value.trim();
  var body = name ? 'Von: ' + name + '\\n\\n' + msg : msg;
  var a = document.createElement('a');
  a.href = 'mailto:samuel.bruelisauer@gmail.com'
         + '?subject=' + encodeURIComponent('Feedback: werstimmtwie')
         + '&body='    + encodeURIComponent(body);
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  document.getElementById('fb').style.display = 'none';
  document.getElementById('ok').style.display = 'block';
});
</script>
""", height=235)

# ════════════════════════════════════════════════════════════════════════════
# 2. THEMA
# ════════════════════════════════════════════════════════════════════════════

elif page == "Thema":
    st.title("Themen")
    st.markdown(
        "Wähle ein politisches Thema und sieh, wie geschlossen die Fraktionen abstimmten "
        "und welche Parlamentsmitglieder am häufigsten von der eigenen Fraktionslinie abwichen. "
        "Die Themen werden anhand von Stichwörtern in Titel und Geschäftsnummern klassifiziert."
    )

    members   = get_members()
    all_votes = get_all_votes()
    if members.empty or all_votes.empty:
        no_data_stop()

    # ── Topic selector ────────────────────────────────────────────────────
    _topic_opts = ["— Thema auswählen —"] + list(TOPICS.keys())
    _topic_sel  = st.selectbox("🔍 Thema wählen", _topic_opts, key="thema_sel")

    # ── Thema × Fraktion cohesion heatmap (only when none selected) ───────
    if _topic_sel == "— Thema auswählen —":
        with st.spinner("Lade Übersicht…"):
            _coh_matrix = get_topic_fraction_cohesion()

        if not _coh_matrix.empty:
            _pivot = _coh_matrix.pivot(index="Thema", columns="Fraktion", values="Kohäsion (%)")
            _col_order = [frac_label(f) for f in FRAC_ORDER if frac_label(f) in _pivot.columns]
            _pivot = _pivot[_col_order]

            _fig_hm = px.imshow(
                _pivot,
                color_continuous_scale="RdYlGn",
                zmin=50, zmax=100,
                text_auto=True,
                aspect="auto",
                labels=dict(color="Kohäsion (%)"),
            )
            _fig_hm.update_layout(
                xaxis_title="",
                yaxis_title="",
                coloraxis_colorbar=dict(title="Kohäsion (%)"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=420,
            )
            st.plotly_chart(_fig_hm, use_container_width=True)
            st.caption(
                "Fraktionskohäsion pro Thema: Grün = Fraktion stimmt geschlossen ab, "
                "Rot = viele Abweichler. Klicke auf ein Feld zum Zoomen."
            )
        st.stop()

    topic = _topic_sel

    with st.spinner("Filtere Abstimmungen nach Thema…"):
        topic_gks = get_topic_groupkeys(all_votes, topic)

    if not topic_gks:
        st.warning("Keine Abstimmungen für dieses Thema gefunden.")
        st.stop()

    st.caption(f"{len(topic_gks)} Abstimmungen zu diesem Thema gefunden.")

    # ── Fraktionskohäsion ─────────────────────────────────────────────────
    st.subheader("Fraktionskohäsion auf diesem Thema")
    topic_votes = all_votes[all_votes["GroupKey"].isin(topic_gks)]
    coh_topic   = fraction_cohesion(topic_votes, members)
    coh_all     = fraction_cohesion(all_votes,   members)

    if not coh_topic.empty:
        coh_topic = coh_topic.rename(columns={"Kohäsion (%)": "Thema"})
        coh_all   = coh_all.rename(columns={"Kohäsion (%)": "Gesamt"})
        coh_merged = coh_topic.merge(coh_all[["Fraktion", "Gesamt"]], on="Fraktion", how="left")
        coh_merged["Fraktion"] = coh_merged["Fraktion"].apply(frac_label)
        coh_long = coh_merged.melt(
            id_vars=["Fraktion", "Mitglieder"],
            value_vars=["Thema", "Gesamt"],
            var_name="Bereich", value_name="Kohäsion (%)",
        )
        fig = px.bar(
            coh_long, x="Fraktion", y="Kohäsion (%)", color="Bereich",
            barmode="group", title=f"Kohäsion: {topic} vs. Gesamt",
            color_discrete_map={"Thema": "#2c3e50", "Gesamt": "#bdc3c7"},
            category_orders={"Fraktion": frac_order_labels()},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Top Abweichler ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Top Abweichler auf diesem Thema")
    positions = get_positions()
    with st.spinner("Berechne Abweichungen…"):
        dev_all = topic_deviators_all(all_votes, members, positions, topic_gks)

    if dev_all.empty:
        st.info("Zu wenig Daten (mind. 3 Abstimmungen pro Mitglied erforderlich).")
    else:
        frac_filter_opts = ["Alle"] + sorted(dev_all["FractionAbbr"].unique())
        frac_filter = st.selectbox("Fraktion filtern", frac_filter_opts, key="thema_frac_filter")
        show = dev_all if frac_filter == "Alle" else dev_all[dev_all["FractionAbbr"] == frac_filter]

        show_plot = show.copy()
        show_plot["Fraktion"] = show_plot["FractionAbbr"].apply(frac_label)
        fig2 = px.bar(
            show_plot.sort_values("Abweichung (%)"),
            x="Abweichung (%)", y="Name", orientation="h",
            color="FractionAbbr",
            color_discrete_map=frac_color_map(),
            hover_data=["Fraktion", "CouncilAbbr", "Kanton", "Abweichend", "Total"],
            custom_data=["PersonNumber"],
            title=f"Abweichungsrate auf Thema: {topic}",
            category_orders={"FractionAbbr": FRAC_ORDER},
            height=max(350, len(show_plot) * 22),
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"},
                           showlegend=False)
        _ev_thema = st.plotly_chart(
            fig2, use_container_width=True,
            key="thema_bar", on_select="rerun",
        )
        if not _mp_click_nav(_ev_thema, "thema_bar", members):
            st.markdown("**Mitglied im Detail ansehen:**")
            _tn = show["Name"].tolist()
            _tp = show["PersonNumber"].tolist()
            _sel = st.selectbox("", _tn, key="thema_nav_sel", label_visibility="collapsed")
            if st.button("→ Zur Mitglied-Seite", key="thema_nav_btn"):
                navigate_to_mp(_tp[_tn.index(_sel)])

    st.markdown("---")
    _thema_breakdown = dev_all[["FractionAbbr", "Abweichung (%)", "Abweichend", "Total"]].to_dict(orient="records") if not dev_all.empty else []
    ai_button("sum_thema", summarizer.summarize_topic, topic, _thema_breakdown)

# ════════════════════════════════════════════════════════════════════════════
# 3. GESCHÄFT
# ════════════════════════════════════════════════════════════════════════════

elif page == "Geschäft":
    st.title("Geschäft")
    members   = get_members()
    all_votes = get_all_votes()
    if members.empty or all_votes.empty:
        no_data_stop()

    _goto = st.session_state.pop("_geschaeft_goto", None)
    if _goto:
        st.session_state["geschaeft_search"] = _goto
    keyword = st.text_input(
        "Suche nach Geschäftsnummer oder Stichwort",
        placeholder="z.B. 23.456  oder  CO2-Gesetz",
        key="geschaeft_search",
    )

    if not keyword.strip():
        # ── Landing: polarity scatter ─────────────────────────────────────
        st.subheader("Lagerwechsel")
        with st.spinner("Berechne Lagerwechsel…"):
            _pol = get_vote_polarity_scatter()
        if not _pol.empty:
            _pol_show = _pol.copy()
            _pol_show["BizTitle"] = _pol_show["BizTitle"].fillna("")
            _pol_show["Subject"]  = _pol_show["Subject"].fillna("")
            _fig_pol = px.scatter(
                _pol_show,
                x="PctDevLeft",
                y="PctDevRight",
                custom_data=["BizTitle", "Subject", "BizShort", "MeetingDate"],
                labels={
                    "PctDevLeft":  "Linksdrift (% der Mitte/FDP/SVP-Mitglieder, die mit G/SP stimmten)",
                    "PctDevRight": "Rechtsdrift (% der G/SP/Mitte-Mitglieder, die mit FDP/SVP stimmten)",
                },
                opacity=0.55,
                color_discrete_sequence=["#5c7caa"],
                height=460,
            )
            _fig_pol.update_traces(
                marker=dict(size=5),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<b>%{customdata[1]}</b><br>"
                    "<b>Geschäft %{customdata[2]}</b><br>"
                    "<br>"
                    "Linksdrift: %{x:.1f}%<br>"
                    "Rechtsdrift: %{y:.1f}%<br>"
                    "%{customdata[3]}"
                    "<extra></extra>"
                ),
            )
            _fig_pol.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor="#ccc"),
                yaxis=dict(showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor="#ccc"),
            )
            _ev_pol = st.plotly_chart(
                _fig_pol, use_container_width=True,
                key="pol_scatter", on_select="rerun",
            )
            st.caption(
                "Jeder Punkt ist eine polarisierte Abstimmung. "
                "**X-Achse**: Anteil der Mitte (GLP, M-E), FDP- und SVP-Mitglieder, "
                "die mit der linken Mehrheit (G/SP) stimmten — je weiter rechts, desto "
                "stärker der Linkszug im nicht-linken Lager. "
                "**Y-Achse**: Anteil der Grünen-, SP- und Mitte-Mitglieder, "
                "die mit der rechten Mehrheit (FDP/SVP) stimmten. "
                "Oben rechts: starke Gegenbewegungen in beiden Lagern. "
                "Punkt anklicken für Direktnavigation."
            )
            _biz_click_nav(_ev_pol, "pol_scatter")

        # ── Landing: direction × magnitude scatter ────────────────────────
        st.markdown("---")
        st.subheader("Richtung & Stärke der Abweichungen")
        with st.spinner("Berechne Richtungsanalyse…"):
            _dm = get_direction_magnitude_scatter()
        if not _dm.empty:
            _COALITION_COLORS = {
                "Parteilinien":      "#888888",
                "Mitte kippt links": "#9B59B6",
                "Mitte kippt rechts":"#1ABC9C",
                "Konsens":           "#BDC3C7",
                "Unheilige Allianz": "#E91E8C",
            }
            _COALITION_ORDER = [
                "Parteilinien", "Mitte kippt links", "Mitte kippt rechts",
                "Konsens", "Unheilige Allianz",
            ]
            _COALITION_DESCS = {
                "Parteilinien":
                    "Standard Links-Rechts-Block: G/SP gegen FDP/SVP. Die Mitte (GLP, M-E) "
                    "ist geteilt oder ohne klare Mehrheit. Hohe Y-Werte bedeuten, dass viele "
                    "einzelne MPs intern gegen ihre Fraktion stimmten — nicht, dass sich Blöcke "
                    "vermischten.",
                "Mitte kippt links":
                    "GLP *und* M-E stimmten geschlossen mit G/SP gegen FDP/SVP. "
                    "Die Linke hatte eine breite parlamentarische Mehrheit.",
                "Mitte kippt rechts":
                    "GLP *und* M-E stimmten geschlossen mit FDP/SVP gegen G/SP. "
                    "Das rechte Lager hatte eine breite parlamentarische Mehrheit.",
                "Konsens":
                    "Mindestens 5 von 6 Fraktionen auf derselben Seite — "
                    "eine seltene, parteiübergreifende Einigkeit.",
                "Unheilige Allianz":
                    "Ideologisch gegensätzliche Fraktionen stimmten zusammen: "
                    "z.B. SP+SVP oder G+FDP — die übliche Lager-Logik ausser Kraft gesetzt.",
            }

            with st.expander("Was bedeuten die Koalitionstypen?"):
                for _ct in _COALITION_ORDER:
                    _col_dot, _col_txt = st.columns([0.04, 0.96])
                    _col_dot.markdown(
                        f"<div style='width:14px;height:14px;border-radius:50%;"
                        f"background:{_COALITION_COLORS[_ct]};margin-top:3px'></div>",
                        unsafe_allow_html=True,
                    )
                    _col_txt.markdown(f"**{_ct}** — {_COALITION_DESCS[_ct]}")

            # Chart + checkboxes side by side
            _chart_col, _chk_col = st.columns([5, 1])

            _dm_active = []
            with _chk_col:
                st.markdown("**Anzeigen**")
                for _ct in _COALITION_ORDER:
                    if st.checkbox(
                        _ct, value=True, key=f"dm_chk_{_ct}",
                    ):
                        _dm_active.append(_ct)

            if not _dm_active:
                _dm_active = list(_COALITION_ORDER)

            _dm_show = _dm[_dm["Koalitionstyp"].isin(_dm_active)].copy()
            _dm_show["BizTitle"] = _dm_show["BizTitle"].fillna("")
            _dm_show["Subject"]  = _dm_show["Subject"].fillna("")

            def _dir_label(x):
                if abs(x) < 0.005:
                    return f"Netto-Richtung = {x:.2f}"
                return (f"← Zug nach links: {abs(x):.2f}"
                        if x < 0 else f"Zug nach rechts →: {x:.2f}")

            _dm_show["_DirLabel"] = _dm_show["NetGravity"].apply(_dir_label)

            with _chart_col:
                _fig_dm = px.scatter(
                    _dm_show,
                    x="NetGravity",
                    y="DeviationPct",
                    color="Koalitionstyp",
                    color_discrete_map=_COALITION_COLORS,
                    category_orders={"Koalitionstyp": _COALITION_ORDER},
                    custom_data=["BizTitle", "Subject", "BizShort",
                                 "_DirLabel", "MeetingDate", "Koalitionstyp"],
                    labels={
                        "NetGravity":   "← Zug nach links · Richtung · Zug nach rechts →",
                        "DeviationPct": "Interne Abweichungsrate (%)",
                    },
                    opacity=0.65,
                    height=480,
                )
                _fig_dm.update_traces(
                    marker=dict(size=5),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "<b>%{customdata[1]}</b><br>"
                        "<b>Geschäft %{customdata[2]}</b><br>"
                        "<br>"
                        "%{customdata[3]}<br>"
                        "Abweichungsrate: %{y:.1f}%<br>"
                        "Koalition: %{customdata[5]}<br>"
                        "%{customdata[4]}"
                        "<extra></extra>"
                    ),
                )
                _fig_dm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(
                        showgrid=True, zeroline=True,
                        zerolinewidth=1.5, zerolinecolor="#aaa",
                        range=[-1.1, 1.1],
                    ),
                    yaxis=dict(showgrid=True, zeroline=False),
                    legend=dict(orientation="v"),
                    legend_title="",
                )
                _ev_dm = st.plotly_chart(
                    _fig_dm, use_container_width=True,
                    key="dm_scatter", on_select="rerun",
                )

            st.caption(
                "Nur polarisierte Abstimmungen (Links- und Rechtslager stimmen unterschiedlich). "
                "**X-Achse**: Netto-Richtung aller Abweichungen — gewichtet nach politischer Distanz "
                "(SVP-Mitglied das links stimmt → starker Linkszug; GLP-Mitglied → schwacher Zug). "
                "**Y-Achse**: Anteil aller Parlamentarier:innen, die gegen ihre *eigene* "
                "Fraktionslinie stimmten — misst interne Disziplin, nicht Blockloyalität. "
                "Punkt anklicken für Direktnavigation."
            )
            _biz_click_nav(_ev_dm, "dm_scatter")
        st.stop()

    with st.spinner("Suche…"):
        results = search_geschaefte(keyword.strip(), get_all_votes())

    if results.empty:
        st.warning("Keine Treffer gefunden.")
        st.stop()

    results["Curia Vista"] = results["BizShort"].apply(_make_curia_url)
    cfg = {"Curia Vista": st.column_config.LinkColumn("Curia Vista", display_text="↗")}
    st.dataframe(results, column_config=cfg, use_container_width=True, height=250)

    # Select a Geschäft for detail
    biz_options = results["BizShort"].tolist()
    biz_titles  = dict(zip(results["BizShort"], results["BizTitle"].fillna("")))
    biz_sel_label = st.selectbox(
        "Geschäft für Detailansicht auswählen",
        [f"{bs}  — {biz_titles.get(bs,'')[:70]}" for bs in biz_options],
        key="geschaeft_detail_sel",
    )
    biz_short = biz_options[
        [f"{bs}  — {biz_titles.get(bs,'')[:70]}" for bs in biz_options].index(biz_sel_label)
    ]

    st.markdown("---")
    biz_url = _make_curia_url(biz_short)
    header_cols = st.columns([3, 1])
    header_cols[0].subheader(f"Geschäft {biz_short}")
    if biz_url:
        header_cols[1].markdown(f"[Curia Vista ↗]({biz_url})")

    with st.spinner("Lade Abstimmungsdaten…"):
        gv = geschaeft_votes_enriched(biz_short, all_votes, members)

    if gv.empty:
        st.warning("Keine Abstimmungsdaten für dieses Geschäft.")
        st.stop()

    # Roll-call selector
    rollcalls = (
        gv.drop_duplicates("GroupKey")
          .sort_values("MeetingDate")[["GroupKey", "MeetingDate", "Subject", "CouncilAbbr"]]
    )
    rc_labels = [
        f"{row['MeetingDate'][:10]}  [{row.get('CouncilAbbr','?')}]  {str(row['Subject'] or '')[:80]}"
        for _, row in rollcalls.iterrows()
    ]
    rc_sel = st.selectbox("Abstimmung auswählen", rc_labels, key="geschaeft_rc_sel")
    sel_gk = rollcalls.iloc[rc_labels.index(rc_sel)]["GroupKey"]

    vote_detail = gv[gv["GroupKey"] == sel_gk].copy()

    # Subject info
    subj = vote_detail["Subject"].iloc[0] if not vote_detail.empty else ""
    if subj:
        st.caption(f"**Frage:** {subj}")

    chambers = vote_detail["CouncilAbbr"].dropna().unique()
    chamber_labels = " & ".join(sorted(chambers))
    st.caption(f"Kammer: {chamber_labels}")

    # MeaningYes / MeaningNo — what does Ja/Nein mean on this specific vote?
    def _first_nonempty(col):
        if col not in vote_detail.columns:
            return ""
        vals = vote_detail[col].dropna()
        vals = vals[vals.astype(str).str.strip() != ""]
        return vals.iloc[0] if not vals.empty else ""

    my = _first_nonempty("MeaningYes")
    mn = _first_nonempty("MeaningNo")
    if my or mn:
        mc1, mc2 = st.columns(2)
        if my:
            mc1.success(f"✅ **Ja** = {my}")
        if mn:
            mc2.error(f"❌ **Nein** = {mn}")

    # Stacked bar per fraction
    active_vote = vote_detail[~vote_detail["DecisionNorm"].isin(ABSENT)].copy()
    if not active_vote.empty:
        active_vote["Fraktion"] = active_vote["FractionAbbr"].apply(frac_label)
        grp = (
            active_vote.groupby(["Fraktion", "DecisionNorm"])
            .size().reset_index(name="Anzahl")
        )
        fig = px.bar(
            grp, x="Fraktion", y="Anzahl", color="DecisionNorm",
            color_discrete_map=COLORS, barmode="stack",
            title="Abstimmung nach Fraktion",
            category_orders={"Fraktion": frac_order_labels()},
        )
        fig.update_layout(legend_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Individual MP vote table
    st.subheader("Einzelstimmen")
    st.caption("🔴 = Abweichler von der Fraktionsmehrheit")

    show_cols = ["Name", "FractionAbbr", "CouncilAbbr", "CantonAbbr",
                 "DecisionNorm", "FracMajority", "Abweichler"]
    show_cols = [c for c in show_cols if c in vote_detail.columns]
    disp = vote_detail[show_cols].rename(columns={
        "FractionAbbr": "Fraktion", "CouncilAbbr": "Kammer",
        "CantonAbbr": "Kanton", "DecisionNorm": "Entscheid",
        "FracMajority": "Fraktionsmehrheit",
    }).sort_values(["Fraktion", "Entscheid"])

    # Highlight deviators
    def _highlight(row):
        if row.get("Abweichler"):
            return ["background-color: #fdecea"] * len(row)
        return [""] * len(row)

    st.dataframe(
        disp.style.apply(_highlight, axis=1),
        use_container_width=True, height=400,
    )

    # Navigation to MP
    st.markdown("**Mitglied im Detail ansehen:**")
    mp_names = vote_detail["Name"].tolist()
    mp_pns   = vote_detail["PersonNumber"].tolist()
    _gsel = st.selectbox("", mp_names, key="geschaeft_mp_sel",
                          label_visibility="collapsed")
    if st.button("→ Zur Mitglied-Seite", key="geschaeft_mp_btn"):
        navigate_to_mp(mp_pns[mp_names.index(_gsel)])

    st.markdown("---")
    _geschaeft_frac_votes = (
        active_vote.groupby(["FractionAbbr", "DecisionNorm"])
        .size().reset_index(name="Anzahl").to_dict(orient="records")
        if not active_vote.empty else []
    )
    ai_button(
        "sum_geschaeft",
        summarizer.summarize_geschaeft,
        biz_short,
        str(subj),
        str(my),
        str(mn),
        _geschaeft_frac_votes,
    )

# ════════════════════════════════════════════════════════════════════════════
# 4. MITGLIED
# ════════════════════════════════════════════════════════════════════════════

elif page == "Mitglied":
    members = get_members()
    if members.empty:
        no_data_stop()

    st.title("Mitglieder des Parlaments")

    # ── Search controls ───────────────────────────────────────────────────
    target_pn = st.session_state.pop("_nav_pn", None)
    if target_pn:
        st.session_state["mp_frac_filter"]   = "Alle"
        st.session_state["mp_canton_filter"] = "Alle"

    all_fracs = sorted(members["FractionAbbr"].dropna().unique())
    _fc1, _fc2 = st.columns(2)
    with _fc1:
        frac_filter = st.selectbox(
            "Fraktion", ["Alle"] + [frac_label(a) for a in all_fracs],
            key="mp_frac_filter",
        )
    with _fc2:
        _filtered_for_canton = members.copy()
        if frac_filter != "Alle":
            for a in all_fracs:
                if frac_label(a) == frac_filter:
                    _filtered_for_canton = members[members["FractionAbbr"] == a]
                    break
        canton_filter = st.selectbox(
            "Kanton",
            ["Alle"] + sorted(_filtered_for_canton["CantonAbbr"].dropna().unique()),
            key="mp_canton_filter",
        )

    filtered = members.copy()
    if frac_filter != "Alle":
        for a in all_fracs:
            if frac_label(a) == frac_filter:
                filtered = filtered[filtered["FractionAbbr"] == a]
                break
    if canton_filter != "Alle":
        filtered = filtered[filtered["CantonAbbr"] == canton_filter]

    if filtered.empty:
        st.warning("Keine Mitglieder für diese Kombination.")
        st.stop()

    _sorted = filtered.sort_values(["LastName", "FirstName"])
    _mp_labels = ["— Name eingeben oder auswählen —"] + [
        member_label(r) for _, r in _sorted.iterrows()
    ]
    _mp_pns = [None] + [int(r["PersonNumber"]) for _, r in _sorted.iterrows()]
    _sel_idx = 0
    if target_pn and int(target_pn) in _mp_pns:
        _sel_idx = _mp_pns.index(int(target_pn))
    _sel_label = st.selectbox(
        "🔍 Mitglied suchen", _mp_labels, index=_sel_idx,
        key="mp_sel", help="Name eintippen zum Filtern",
    )
    pn = _mp_pns[_mp_labels.index(_sel_label)]

    # ── MP × Fraktion alignment heatmap (only when none selected) ────────
    if pn is None:
      with st.spinner("Berechne Übereinstimmungsmatrix…"):
        _aln = get_mp_fraction_alignment()

      if not _aln.empty:
        from plotly.subplots import make_subplots
        _aln_plot = _aln[_aln["FractionAbbr"].isin(FRAC_ORDER)].copy()
        _aln_sorted = pd.concat([
            _aln_plot[_aln_plot["FractionAbbr"] == f].sort_values(f, ascending=False)
            for f in FRAC_ORDER if f in _aln_plot["FractionAbbr"].values
        ])
        _frac_cols_present = [f for f in FRAC_ORDER if f in _aln_sorted.columns]
        _z = _aln_sorted[_frac_cols_present].values
        _ylabels = _aln_sorted["Name"].tolist()
        _xlabels = [frac_label(f) for f in _frac_cols_present]
        _n_mps = len(_aln_sorted)

        # Use integer y-axis so separator lines live in data coords and zoom correctly
        _yints = list(range(_n_mps))

        # Discrete colorscale for the faction strip
        _frac_to_idx = {f: i for i, f in enumerate(FRAC_ORDER)}
        _frac_z = [[float(_frac_to_idx.get(f, 0))] for f in _aln_sorted["FractionAbbr"]]
        _n_fracs = len(FRAC_ORDER)
        _frac_cs = []
        for _i, _f in enumerate(FRAC_ORDER):
            _vlo = _i / _n_fracs
            _vhi = min((_i + 0.99) / _n_fracs, 1.0)
            _frac_cs += [[_vlo, FRAC_COLORS[_f]], [_vhi, FRAC_COLORS[_f]]]
        _frac_cs.append([1.0, FRAC_COLORS[FRAC_ORDER[-1]]])

        # Separator lines at exact half-integer positions between faction groups
        _shapes = []
        _r = 0
        for _frac in FRAC_ORDER:
            _nf = int((_aln_sorted["FractionAbbr"] == _frac).sum())
            if _nf == 0:
                continue
            if _r > 0:
                _shapes.append(dict(
                    type="line", xref="paper", x0=0, x1=1,
                    yref="y", y0=_r - 0.5, y1=_r - 0.5,
                    line=dict(color="white", width=2), layer="above",
                ))
            _r += _nf

        _height = _n_mps * 14 + 60  # 14 px per row + 50 top margin + 10 bottom

        # Colour legend rendered as HTML above the chart (Plotly colorbar y-coords
        # proved unreliable in make_subplots context across Plotly 6 / Streamlit)
        st.markdown(
            "<div style='display:flex;align-items:center;gap:8px;"
            "margin-left:138px;margin-bottom:4px;'>"
            "<span style='font-size:11px;white-space:nowrap;'>30%</span>"
            "<div style='background:linear-gradient(to right,"
            "#d73027,#fc8d59,#fee08b,#ffffbf,#d9ef8b,#91cf60,#1a9850);"
            "height:12px;width:260px;border-radius:2px;flex-shrink:0;'></div>"
            "<span style='font-size:11px;white-space:nowrap;'>100%</span>"
            "<span style='font-size:11px;color:#888;margin-left:6px;'>"
            "Übereinstimmung (%)</span></div>",
            unsafe_allow_html=True,
        )

        _fig_aln = make_subplots(
            rows=1, cols=2,
            column_widths=[0.025, 0.975],
            shared_yaxes=True,
            horizontal_spacing=0.0,
        )
        # Left strip: faction colour
        _fig_aln.add_trace(go.Heatmap(
            z=_frac_z, x=[""], y=_yints,
            colorscale=_frac_cs, zmin=0, zmax=_n_fracs,
            showscale=False, ygap=0,
            texttemplate="",
            customdata=[[lb] for lb in _ylabels],
            hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
        ), row=1, col=1)
        # Right panel: alignment matrix — replace NaN with None so cells stay blank
        _z_clean = [[None if (isinstance(v, float) and v != v) else v for v in row]
                    for row in _z.tolist()]
        _fig_aln.add_trace(go.Heatmap(
            z=_z_clean, x=_xlabels, y=_yints,
            colorscale="RdYlGn", zmin=30, zmax=100,
            showscale=False,
            xgap=1, ygap=0,
            texttemplate="",
            customdata=[[lb] * len(_frac_cols_present) for lb in _ylabels],
            hovertemplate="<b>%{customdata}</b><br>%{x}: %{z:.0f}%<extra></extra>",
        ), row=1, col=2)

        _shared_yaxis = dict(
            autorange="reversed",
            tickmode="array",
            tickvals=_yints,
            ticktext=_ylabels,
            tickfont=dict(size=9),
        )
        _fig_aln.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=_shared_yaxis,
            xaxis=dict(showticklabels=False),
            xaxis2=dict(side="top", tickfont=dict(size=12)),
            margin=dict(l=130, r=20, t=50, b=10),
            height=_height,
            shapes=_shapes,
        )
        st.plotly_chart(_fig_aln, use_container_width=True)
        st.caption(
            "Jede Zeile = ein Parlamentsmitglied, sortiert nach Fraktion und Übereinstimmung mit der eigenen Fraktionsmehrheit. "
            "Der farbige Streifen links zeigt die Fraktionszugehörigkeit (gleiche Farben wie in anderen Grafiken). "
            "Grün = häufig mit dieser Fraktion gestimmt, Rot = selten. Hover für genaue Werte."
        )

    if pn is None:
        st.stop()

    # ── MP profile ────────────────────────────────────────────────────────
    mp_row  = members[members["PersonNumber"] == pn].iloc[0]
    mp_name = f"{mp_row['FirstName']} {mp_row['LastName']}"
    frac    = mp_row["FractionAbbr"]

    st.markdown("---")
    st.title(mp_name)
    st.caption(
        f"{mp_row['CouncilAbbr']} · {frac_label(frac)} · Kanton {mp_row['CantonAbbr']}"
    )

    with st.spinner("Lade Abstimmungen…"):
        mp_votes = get_mp_votes(pn)

    if mp_votes.empty:
        st.warning("Keine Abstimmungsdaten für dieses Mitglied.")
        st.stop()

    # ── Kennzahlen ────────────────────────────────────────────────────────
    active_mask = ~mp_votes["DecisionNorm"].isin(ABSENT)
    absent_mask = mp_votes["DecisionNorm"].isin(
        {"Abwesend", "Entschuldigt", "Nicht teilgenommen"}
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Abstimmungen total", len(mp_votes))
    c2.metric("Aktiv abgestimmt", active_mask.sum())
    c3.metric("Abwesend / Entschuldigt", absent_mask.sum())
    c4.metric("Präsenz", f"{round(active_mask.mean()*100)}%")

    # ── Abweichungsanalyse ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Abweichung von {frac_label(frac)}")

    with st.spinner("Berechne Abweichungen…"):
        all_votes  = get_all_votes()
        frac_votes = get_fraction_votes(frac)
        combined   = pd.concat([frac_votes, mp_votes]).drop_duplicates(
            subset=["GroupKey", "PersonNumber"])
        dev_df = mp_deviation(pn, combined, members)

    if dev_df.empty:
        st.info("Keine Abweichungsdaten verfügbar.")
    else:
        score = deviation_score(dev_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Abweichungsrate", f"{score['pct']}%",
                  help="Anteil aktiver Abstimmungen, bei denen anders als Fraktionsmehrheit votiert wurde")
        c2.metric("Abweichend", score["deviated"])
        c3.metric("Total aktiv", score["total"])

        # Per-Thema deviation
        topic_dev = topic_deviation(pn, combined, members)
        if not topic_dev.empty:
            fig = px.bar(
                topic_dev, x="Thema", y="Abweichung (%)",
                color="Abweichung (%)", color_continuous_scale="RdYlGn_r",
                hover_data=["Abweichend", "Total"],
                title="Abweichungsrate nach Thema",
            )
            fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Per-Session data (computed here, rendered below before Kammernvergleich)
        active_df = dev_df[~dev_df["DecisionNorm"].isin(ABSENT)].copy()
        _sess_chart_df = None
        if not active_df.empty and "SessionId" in active_df.columns:
            sess_df   = get_sessions()
            sess_map  = dict(zip(sess_df["ID"].astype(str), sess_df["Name"]))
            sess_order = sess_df["Name"].tolist()
            active_df["Session"] = active_df["SessionId"].astype(str).map(sess_map).fillna("?")
            by_sess = (
                active_df.groupby("Session")["Deviated"].mean() * 100
            ).round(1).reset_index()
            by_sess.columns = ["Session", "Abweichung (%)"]
            by_sess["_ord"] = by_sess["Session"].map(
                {n: i for i, n in enumerate(sess_order)}
            )
            by_sess = by_sess.sort_values("_ord").drop(columns=["_ord"])
            if len(by_sess) > 1:
                _sess_chart_df = by_sess

        # Deviation direction
        st.subheader("In welche Richtung?")
        st.caption(
            "Wenn dieses Mitglied von der Fraktion abweicht — "
            "mit welcher anderen Fraktion stimmt es dann überein?"
        )
        deviated = dev_df[dev_df["Deviated"]].copy()
        if not deviated.empty:
            positions = get_positions()
            alignment = deviation_alignment_fast(deviated, frac, positions)
            if not alignment.empty:
                align_plot = alignment.copy()
                align_plot["Fraktion"] = align_plot["Fraktion"].apply(frac_label)
                fig3 = px.bar(
                    align_plot, x="Anteil (%)", y="Fraktion",
                    orientation="h", color="Fraktion",
                    color_discrete_map=frac_color_map(),
                    title="Übereinstimmung mit anderen Fraktionen bei Abweichungen",
                    hover_data=["Übereinstimmungen"],
                )
                fig3.update_layout(
                    yaxis={"categoryorder": "array",
                           "categoryarray": list(reversed(frac_order_labels()))},
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Zu wenig gemeinsame Abstimmungen für Richtungsanalyse.")

            # Deviated votes table with filters
            st.subheader("Abweichende Abstimmungen")
            frac_strength = fraction_majority_strength(combined, members, frac)
            deviated["Fraktionsmehrheit"] = deviated["GroupKey"].map(frac_strength).fillna("—")

            _all_topics = sorted(TOPICS.keys())
            _other_fracs = [f for f in FRAC_ORDER if f != frac and f in positions.columns]

            # Staging flag: reset all topic checkboxes to True before the form renders
            if st.session_state.pop("_dev_topic_reset", False):
                for _ci in range(len(_all_topics)):
                    st.session_state[f"dev_topic_{_ci}"] = True

            # Reuse topic_dev (already computed above) — same metric as "Abweichungsrate nach Thema"
            _topic_share = {row["Thema"]: row["Abweichung (%)"] for _, row in topic_dev.iterrows()}
            for _t in _all_topics:
                _topic_share.setdefault(_t, 0.0)
            _sorted_topics = sorted(_all_topics, key=lambda t: _topic_share[t], reverse=True)
            _alpha_idx = {t: i for i, t in enumerate(_all_topics)}

            # Streamlit color names closest to each party's brand color
            _frac_md_color = {
                "RL": "blue", "S": "red", "V": "green",
                "G": "green", "GL": "violet", "M-E": "orange", "BD": "gray",
            }

            with st.form("dev_filters"):
                _tc1, _tc2, _tc3, _tfc = st.columns([3, 3, 3, 2])
                _topic_sel = []
                for _col, _chunk in zip([_tc1, _tc2, _tc3],
                                        [_sorted_topics[0:5], _sorted_topics[5:10],
                                         _sorted_topics[10:14]]):
                    for _t in _chunk:
                        _lbl = f"{TOPIC_EMOJI.get(_t,'')} {_t} ({_topic_share[_t]:.1f}%)"
                        if _col.checkbox(_lbl, value=True, key=f"dev_topic_{_alpha_idx[_t]}"):
                            _topic_sel.append(_t)
                with _tfc:
                    st.markdown("**Mit Fraktion**")
                    _frac_sel = []
                    for _f in _other_fracs:
                        _c = _frac_md_color.get(_f, "")
                        _lbl = f":{_c}[{frac_label(_f)}]" if _c else frac_label(_f)
                        if st.checkbox(_lbl, value=True, key=f"dev_frac_form_{_f}"):
                            _frac_sel.append(_f)
                _btn1, _btn2 = st.columns([1, 2])
                _btn1.form_submit_button("Anwenden")
                _cleared = _btn2.form_submit_button("× Themen zurücksetzen")

            if _cleared:
                st.session_state["_dev_topic_reset"] = True
                st.rerun()

            _dev_show = deviated.copy()
            if _topic_sel and len(_topic_sel) < len(_all_topics):
                _tagged   = tag_topics(_dev_show)
                _dev_show = _tagged[_tagged["Topics"].map(
                    lambda ts: bool(set(ts) & set(_topic_sel))
                )]
            if 0 < len(_frac_sel) < len(_other_fracs):
                _frac_mask = pd.Series(False, index=_dev_show.index)
                for _sf in _frac_sel:
                    if _sf in positions.columns:
                        _maj = _dev_show["GroupKey"].map(positions[_sf])
                        _frac_mask = _frac_mask | (_dev_show["DecisionNorm"] == _maj)
                _dev_show = _dev_show[_frac_mask]

            # Inline table with row-click → Geschäft navigation
            _dv_disp = _dev_show.copy()
            _dv_disp["🔗"] = _dv_disp.apply(
                lambda r: _make_curia_url(r.get("BizShort",""), r.get("BizNumber","")), axis=1
            )
            _dv_cols = [c for c in ["MeetingDate","BizShort","Subject","BizTitle",
                                     "DecisionNorm","Fraktionsmehrheit","🔗"]
                        if c in _dv_disp.columns]
            _dv_disp = (
                _dv_disp[_dv_cols]
                .rename(columns={"MeetingDate":"Datum","BizShort":"Geschäft",
                                  "Subject":"Frage","BizTitle":"Titel",
                                  "DecisionNorm":"Entscheid"})
                .sort_values("Datum", ascending=False)
                .reset_index(drop=True)
            )
            _dv_ev = st.dataframe(
                _dv_disp,
                column_config={"🔗": st.column_config.LinkColumn("Curia Vista", display_text="↗")},
                use_container_width=True, height=400,
                on_select="rerun", selection_mode="single-row",
                key="dev_table_sel",
            )
            if _dv_ev.selection.rows:
                _sel_biz = str(_dv_disp.iloc[_dv_ev.selection.rows[0]]["Geschäft"])
                _href_biz = "?" + urlencode({"page": "Geschäft", "q": _sel_biz})
                st.markdown(
                    f'<a href="{_href_biz}" target="_self" style="display:inline-block;'
                    f'padding:0.35rem 0.7rem;background:#f63366;color:white;'
                    f'border-radius:0.3rem;text-decoration:none;font-size:0.875rem;'
                    f'font-weight:500;">→ Geschäft {_sel_biz}</a>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Keine Abweichungen gefunden.")

        # AI summary
        st.markdown("---")
        dev_dicts = deviated.head(25).to_dict("records") if not deviated.empty else []
        td_dicts  = topic_dev.to_dict("records") if not topic_dev.empty else []
        if dev_dicts:
            ai_button("sum_mp", summarizer.summarize_deviation_votes,
                      mp_name, frac, dev_dicts)
        else:
            ai_button("sum_mp_p", summarizer.summarize_mp_profile,
                      mp_name, frac, score, td_dicts)

        # Session timeline (moved here, just before Kammernvergleich)
        if _sess_chart_df is not None:
            st.markdown("---")
            fig2 = px.line(_sess_chart_df, x="Session", y="Abweichung (%)",
                           title="Abweichungsrate pro Session", markers=True)
            fig2.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Kammernvergleich ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Kammernvergleich")
    st.caption(
        "Wie stark stimmt dieses Mitglied mit der eigenen Fraktion in beiden Kammern überein?"
    )
    with st.spinner("Berechne Kammernvergleich…"):
        cc = cross_chamber_mp(pn, get_all_votes(), members)

    if not cc or not cc.get("vs_oth"):
        st.info(
            f"Kein Kammernvergleich möglich — {frac_label(frac)} "
            f"hat keine Mitglieder in der anderen Kammer mit gemeinsamen Geschäften. "
            f"(SR-Daten werden schrittweise via XLS-Import ergänzt.)"
        )
    else:
        own_lbl = f"{frac_label(cc['frac'])}-{cc['own_council']}"
        oth_lbl = f"{frac_label(cc['frac'])}-{cc['oth_council']}"
        kc1, kc2, kc3 = st.columns(3)
        if cc.get("vs_own"):
            kc1.metric(
                f"Übereinstimmung mit {own_lbl}",
                f"{cc['vs_own']['pct']}%",
                help=f"Auf {cc['vs_own']['total']} gemeinsamen Geschäften",
            )
        if cc.get("vs_oth"):
            kc2.metric(
                f"Übereinstimmung mit {oth_lbl}",
                f"{cc['vs_oth']['pct']}%",
                help=f"Auf {cc['vs_oth']['total']} gemeinsamen Geschäften",
            )
        if cc.get("chambers_agree"):
            kc3.metric(
                f"Einigkeit {own_lbl} vs {oth_lbl}",
                f"{cc['chambers_agree']['pct']}%",
                help=f"Auf {cc['chambers_agree']['total']} gemeinsamen Geschäften",
            )
        if cc.get("vs_own") and cc.get("vs_oth"):
            diff = cc["vs_own"]["pct"] - cc["vs_oth"]["pct"]
            if diff > 5:
                st.caption(
                    f"➡ Dieses Mitglied stimmt stärker mit der {own_lbl} "
                    f"als mit der {oth_lbl} überein (+{diff:.1f} Pp)."
                )
            elif diff < -5:
                st.caption(
                    f"➡ Dieses Mitglied stimmt stärker mit der {oth_lbl} "
                    f"als mit der {own_lbl} überein ({diff:.1f} Pp)."
                )

    # ── Full vote history ─────────────────────────────────────────────────
    with st.expander("Alle Abstimmungen"):
        topic_f = st.selectbox("Thema", ["Alle"] + list(TOPICS.keys()), key="mp_full_topic")
        show = (
            filter_by_topic(tag_topics(mp_votes), topic_f)
            if topic_f != "Alle" else mp_votes
        )
        vote_table(show, height=500)

# ════════════════════════════════════════════════════════════════════════════
# 5. FRAKTION
# ════════════════════════════════════════════════════════════════════════════

elif page == "Fraktion":
    members = get_members()
    if members.empty:
        no_data_stop()

    st.title("Fraktionen")

    # ── Fraction selector ─────────────────────────────────────────────────
    _frac_opts_ordered = [f for f in FRAC_ORDER if f in members["FractionAbbr"].values]
    _frac_other = sorted([f for f in members["FractionAbbr"].dropna().unique()
                          if f not in FRAC_ORDER])
    _all_frac_abbrs = _frac_opts_ordered + _frac_other
    _frac_sel_labels = ["— Fraktion auswählen —"] + [frac_label(a) for a in _all_frac_abbrs]

    _frac_sel = st.selectbox("🔍 Fraktion auswählen", _frac_sel_labels, key="frac_sel")

    # ── Landing: Kohäsion aller Fraktionen (only when none selected) ──────
    if _frac_sel == "— Fraktion auswählen —":
        with st.spinner("Berechne Kohäsion…"):
            all_votes = get_all_votes()
            coh_all   = fraction_cohesion(all_votes, members)

        if not coh_all.empty:
            _coh_plot = coh_all[coh_all["Fraktion"].isin(FRAC_ORDER)].copy()
            _coh_plot["FraktionLabel"] = _coh_plot["Fraktion"].apply(frac_label)
            _fig_coh = px.bar(
                _coh_plot,
                x="FraktionLabel", y="Kohäsion (%)",
                color="FraktionLabel",
                color_discrete_map={frac_label(a): FRAC_COLORS[a] for a in FRAC_ORDER},
                text="Kohäsion (%)",
                hover_data={"FraktionLabel": False, "Mitglieder": True, "Kohäsion (%)": True},
                category_orders={"FraktionLabel": frac_order_labels()},
            )
            _fig_coh.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                textfont=dict(size=13),
            )
            _fig_coh.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis=dict(title="Kohäsion (%)", range=[0, 105]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(_fig_coh, use_container_width=True)
            st.caption(
                "Anteil der Abstimmungen, bei denen alle anwesenden Fraktionsmitglieder gleich stimmten. "
                "Je höher, desto disziplinierter die Fraktion."
            )

        # ── Deviation strip chart ─────────────────────────────────────────
        with st.spinner("Lade Abweichungsübersicht…"):
            _dev_mp = get_all_mp_deviations()

        if not _dev_mp.empty:
            _strip_mp = _dev_mp[_dev_mp["FractionAbbr"].isin(FRAC_ORDER)].copy()
            _strip_mp["Fraktion"] = _strip_mp["FractionAbbr"].apply(frac_label)
            _fig_mp_strip = px.strip(
                _strip_mp, x="Abweichung (%)", y="Fraktion",
                color="FractionAbbr", color_discrete_map=frac_color_map(),
                hover_name="Name",
                hover_data={"FractionAbbr": False, "Fraktion": True,
                            "CouncilAbbr": True, "CantonAbbr": True,
                            "Total": True, "Deviated": True, "Abweichung (%)": True},
                category_orders={"Fraktion": list(reversed(frac_order_labels()))},
                custom_data=["PersonNumber"],
                stripmode="overlay",
                labels={"CouncilAbbr": "Rat", "CantonAbbr": "Kanton",
                        "Total": "Abstimmungen", "Deviated": "Abweichungen"},
            )
            _fig_mp_strip.update_traces(
                marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="white")),
                jitter=0.6,
            )
            _fig_mp_strip.update_layout(
                showlegend=False,
                xaxis=dict(title="Abweichung vom Fraktionskurs (%)",
                           range=[-1, _strip_mp["Abweichung (%)"].max() + 2]),
                yaxis_title="", height=360, margin=dict(l=10, r=10, t=20, b=10),
            )
            _ev_strip = st.plotly_chart(
                _fig_mp_strip, use_container_width=True,
                key="frac_strip", on_select="rerun",
            )
            st.caption("Jeder Punkt ist ein Parlamentsmitglied. Punkt anklicken für Direktnavigation.")
            _mp_click_nav(_ev_strip, "frac_strip", members)


        st.stop()

    with st.spinner("Berechne Kohäsion…"):
        all_votes = get_all_votes()
        coh_all   = fraction_cohesion(all_votes, members)

    frac      = _all_frac_abbrs[_frac_sel_labels.index(_frac_sel) - 1]
    frac_name = members.loc[members["FractionAbbr"] == frac, "FractionName"].iloc[0]

    st.markdown("---")
    st.title(frac_label(frac))
    st.caption(frac_name)

    with st.spinner("Lade Daten…"):
        frac_votes = get_fraction_votes(frac)

    if frac_votes.empty:
        st.warning("Keine Daten für diese Fraktion.")
        st.stop()

    # ── Kohäsion dieser Fraktion ──────────────────────────────────────────
    st.subheader("Fraktionskohäsion")
    if not coh_all.empty:
        this_coh = coh_all.loc[coh_all["Fraktion"] == frac, "Kohäsion (%)"]
        if not this_coh.empty:
            st.metric(f"Kohäsion {frac_label(frac)}", f"{this_coh.iloc[0]}%")

    # ── Abweichler ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Wer weicht am stärksten ab?")
    with st.spinner("Berechne Abweichungen…"):
        dev_table = fraction_all_deviations(frac, frac_votes, members)

    if dev_table.empty:
        st.info("Keine Abweichungsdaten.")
    else:
        fig2 = px.bar(
            dev_table.sort_values("Abweichung (%)"),
            x="Abweichung (%)", y="Name", orientation="h",
            color="Abweichung (%)", color_continuous_scale="RdYlGn_r",
            hover_data=["Kanton", "Abweichend", "Total"],
            title=f"Abweichungsraten — {frac_label(frac)}",
            height=max(350, len(dev_table) * 26),
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"},
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(dev_table.drop(columns=["PersonNumber"]),
                     use_container_width=True, height=300)

        st.markdown("**Mitglied im Detail ansehen:**")
        _fn = dev_table["Name"].tolist()
        _fp = dev_table["PersonNumber"].tolist()
        _fs = st.selectbox("", _fn, key="frac_nav_sel", label_visibility="collapsed")
        if st.button("→ Zur Mitglied-Seite", key="frac_nav_btn"):
            navigate_to_mp(_fp[_fn.index(_fs)])

        ai_button("sum_frac", summarizer.summarize_fraction_deviators,
                  frac, dev_table.to_dict("records"))

    # ── Richtungs-Heatmap ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("In welche Richtung weichen sie ab?")
    with st.spinner("Berechne Richtungen…"):
        positions   = get_positions()
        heatmap_df  = fraction_direction_heatmap(frac, all_votes, members, positions)

    if heatmap_df.empty:
        st.info("Zu wenig Daten für Heatmap (min. 5 Abweichungen pro Mitglied).")
    else:
        if not dev_table.empty:
            order = dev_table["Name"].tolist()
            heatmap_df = heatmap_df.reindex(
                [n for n in order if n in heatmap_df.index]
            )
        heatmap_labeled = heatmap_df.rename(columns=frac_label)
        col_order = [c for c in frac_order_labels() if c in heatmap_labeled.columns]
        heatmap_labeled = heatmap_labeled.reindex(columns=col_order)
        fig3 = px.imshow(
            heatmap_labeled,
            color_continuous_scale="Blues", aspect="auto",
            title=f"Abweichungsrichtung — {frac_label(frac)} (% Abweichungen aligned mit Fraktion X)",
            labels={"color": "%", "x": "Andere Fraktion", "y": "Mitglied"},
        )
        fig3.update_layout(
            height=len(heatmap_df) * 36 + 60,
            yaxis=dict(dtick=1),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Kammernvergleich ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Kammernvergleich NR vs SR")
    with st.spinner("Berechne Kammernvergleich…"):
        cc_df = cross_chamber_fraction(frac, all_votes, members)

    if cc_df.empty:
        st.info(
            f"{frac_label(frac)} hat keine Mitglieder in beiden Kammern "
            "mit gemeinsamen Abstimmungsgeschäften — kein Vergleich möglich. "
            "(SR-Daten werden schrittweise via XLS-Import ergänzt.)"
        )
    else:
        n_total  = len(cc_df)
        n_einig  = int(cc_df["Einig"].sum())
        pct_einig = round(n_einig / n_total * 100, 1) if n_total else 0
        kc1, kc2 = st.columns(2)
        kc1.metric("Gemeinsame Geschäfte", n_total)
        kc2.metric("NR & SR einig", f"{pct_einig}%")

        disagree = cc_df[~cc_df["Einig"]].copy()
        if disagree.empty:
            st.success("NR und SR stimmten auf allen gemeinsamen Geschäften gleich!")
        else:
            st.caption(f"{len(disagree)} Geschäfte mit abweichender Position:")
            disag_disp = disagree[["BizShort", "BizTitle", "NR", "SR", "Datum"]].rename(
                columns={"BizShort": "Geschäft", "BizTitle": "Titel"}
            )
            disag_disp["Curia Vista"] = disag_disp["Geschäft"].apply(_make_curia_url)
            cfg = {"Curia Vista": st.column_config.LinkColumn("Curia Vista", display_text="↗")}
            st.dataframe(disag_disp, column_config=cfg,
                         use_container_width=True, height=300)

