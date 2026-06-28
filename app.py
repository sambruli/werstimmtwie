"""
Swiss Parliament Voting Analyser — 7-page Streamlit app.
Run: streamlit run app.py
"""

import os
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
    name = FRAC_NAMES.get(str(abbr), "")
    return f"{abbr} ({name})" if name else str(abbr)

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
               get_sunburst_data):
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
    st.session_state["_nav_page"] = "Mitglied"
    st.session_state["_nav_pn"]   = int(pn)
    st.rerun()

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

_pages = ["Übersicht", "Thema", "Geschäft", "Mitglied", "Fraktion"]
_nav = st.session_state.pop("_nav_page", None)
if _nav and _nav in _pages:
    st.session_state["nav_radio"] = _nav
page = st.sidebar.radio("Navigation", _pages, key="nav_radio", label_visibility="collapsed")

# ════════════════════════════════════════════════════════════════════════════
# 1. ÜBERSICHT
# ════════════════════════════════════════════════════════════════════════════

if page == "Übersicht":
    st.title("Wer stimmt wie?")
    st.markdown("##### Abstimmungsanalyse Schweizer Parlamentarier:innen")

    st.markdown(
        "Diese App analysiert das Abstimmungsverhalten aller Mitglieder des Schweizer Parlaments "
        "(National- und Ständerat, Legislatur 52 ab Dezember 2023). "
        "Im Zentrum steht die Frage: Wer weicht wie oft von der Fraktionslinie ab — "
        "bei welchen Themen, in welche Richtung, und im Vergleich zu anderen Fraktionen?"
    )

    # ── Sunburst ──────────────────────────────────────────────────────────
    with st.spinner("Lade Übersicht…"):
        _sb_ids, _sb_labels, _sb_parents, _sb_values, _sb_colors = get_sunburst_data()

    if _sb_ids:
        _fig_sb = go.Figure(go.Sunburst(
            ids=_sb_ids, labels=_sb_labels, parents=_sb_parents,
            values=_sb_values,
            marker=dict(colors=_sb_colors),
            branchvalues="total",
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>%{value} Stimmen<extra></extra>",
            textfont=dict(size=11),
        ))
        _fig_sb.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=520,
        )
        st.plotly_chart(_fig_sb, use_container_width=True)
        st.caption(
            "Innerer Ring: Fraktionen. Mittlerer Ring: Parlamentsmitglieder. "
            "Äusserer Ring: Abstimmungen im Einklang mit der Fraktion (Farbe) vs. abweichend (grau). "
            "Klicken zum Hineinzoomen."
        )

    st.markdown("---")
    st.subheader("So verwendest du die App")
    st.markdown(
        "**Thema** — Wähle ein politisches Thema (z.B. Klima, Migration) und sieh, "
        "wie kohäsiv die Fraktionen abstimmten und wer am stärksten von der eigenen Linie abwich.\n\n"
        "**Geschäft** — Suche nach einer Geschäftsnummer (z.B. `23.456`) oder einem Stichwort "
        "und sieh, wie jede Fraktion bei jeder einzelnen Abstimmung votierte.\n\n"
        "**Mitglied** — Wähle ein Parlamentsmitglied und erhalte ein detailliertes Profil: "
        "Gesamtabweichung, Abweichung nach Thema, und eine Liste aller abweichenden Abstimmungen.\n\n"
        "**Fraktion** — Vergleiche Kohäsion und Abweichungsraten innerhalb einer Fraktion, "
        "inklusive Heatmap über alle Abstimmungen."
    )

    st.warning(
        "**🤖 KI-Zusammenfassungen** — Jede Seite bietet eine KI-Zusammenfassung via Claude. "
        "Bitte diese Funktion sparsam benutzen: die Kosten werden von Sam's persönlichem API-Budget gedeckt."
    )

    st.markdown("---")
    st.subheader("Methodik")
    st.info(_METHODOLOGY)

    st.markdown("---")
    st.caption("📊 *Wähle eine Seite in der Navigation, um die Analyse zu starten.*")

# ════════════════════════════════════════════════════════════════════════════
# 2. THEMA
# ════════════════════════════════════════════════════════════════════════════

elif page == "Thema":
    st.title("Suche nach Thema")
    st.markdown(
        "Wähle ein politisches Thema und sieh, wie geschlossen die Fraktionen abstimmten "
        "und welche Parlamentsmitglieder am häufigsten von der eigenen Fraktionslinie abwichen. "
        "Die Themen werden anhand von Stichwörtern in Titel und Geschäftsnummern klassifiziert."
    )

    members   = get_members()
    all_votes = get_all_votes()
    if members.empty or all_votes.empty:
        no_data_stop()

    # ── Option 3: Thema × Fraktion cohesion heatmap (welcome visual) ──────
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

    st.markdown("---")

    # ── Topic selector ────────────────────────────────────────────────────
    _topic_opts = ["— Thema auswählen —"] + list(TOPICS.keys())
    _topic_sel  = st.selectbox("🔍 Thema wählen", _topic_opts, key="thema_sel")

    if _topic_sel == "— Thema auswählen —":
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
            title=f"Abweichungsrate auf Thema: {topic}",
            category_orders={"FractionAbbr": FRAC_ORDER},
            height=max(350, len(show_plot) * 22),
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"},
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

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
    st.title("Geschäft-Suche")
    members   = get_members()
    all_votes = get_all_votes()
    if members.empty or all_votes.empty:
        no_data_stop()

    keyword = st.text_input(
        "Suche nach Geschäftsnummer oder Stichwort",
        placeholder="z.B. 23.456  oder  CO2-Gesetz",
        key="geschaeft_search",
    )

    if not keyword.strip():
        st.info("Gib eine Geschäftsnummer (z.B. 24.012) oder ein Stichwort ein.")
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

    st.title("Suche nach Mitgliedern des Parlaments")

    # ── Deviation strip chart (welcome visual) ────────────────────────────
    with st.spinner("Lade Übersicht…"):
        _dev_mp = get_all_mp_deviations()

    if not _dev_mp.empty:
        _strip_mp = _dev_mp[_dev_mp["FractionAbbr"].isin(FRAC_ORDER)].copy()
        _strip_mp["Fraktion"] = _strip_mp["FractionAbbr"].apply(frac_label)

        _fig_mp_strip = px.strip(
            _strip_mp,
            x="Abweichung (%)",
            y="Fraktion",
            color="FractionAbbr",
            color_discrete_map=frac_color_map(),
            hover_name="Name",
            hover_data={
                "FractionAbbr": False,
                "Fraktion": True,
                "CouncilAbbr": True,
                "CantonAbbr": True,
                "Total": True,
                "Deviated": True,
                "Abweichung (%)": True,
            },
            category_orders={"Fraktion": list(reversed(frac_order_labels()))},
            stripmode="overlay",
            labels={
                "CouncilAbbr": "Rat",
                "CantonAbbr": "Kanton",
                "Total": "Abstimmungen",
                "Deviated": "Abweichungen",
            },
        )
        _fig_mp_strip.update_traces(
            marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="white")),
            jitter=0.6,
        )
        _fig_mp_strip.update_layout(
            showlegend=False,
            xaxis=dict(
                title="Abweichung vom Fraktionskurs (%)",
                range=[-1, _strip_mp["Abweichung (%)"].max() + 2],
            ),
            yaxis_title="",
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(_fig_mp_strip, use_container_width=True)

    st.markdown("---")

    # ── Search controls ───────────────────────────────────────────────────
    # Handle cross-page navigation (e.g. from Geschäft page)
    target_pn = st.session_state.pop("_nav_pn", None)
    if target_pn:
        st.session_state["mp_frac_filter"]   = "Alle"
        st.session_state["mp_canton_filter"] = "Alle"

    # Fraction + Canton filters on one row
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

    # Apply filters
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

    # Name search — full width, with placeholder (no default selection)
    _sorted = filtered.sort_values(["LastName", "FirstName"])
    _mp_labels = ["— Name eingeben oder auswählen —"] + [
        member_label(r) for _, r in _sorted.iterrows()
    ]
    _mp_pns = [None] + [int(r["PersonNumber"]) for _, r in _sorted.iterrows()]

    _sel_idx = 0
    if target_pn and int(target_pn) in _mp_pns:
        _sel_idx = _mp_pns.index(int(target_pn))

    _sel_label = st.selectbox(
        "🔍 Mitglied suchen",
        _mp_labels,
        index=_sel_idx,
        key="mp_sel",
        help="Name eintippen zum Filtern",
    )
    pn = _mp_pns[_mp_labels.index(_sel_label)]

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

        # Per-Session timeline
        active_df = dev_df[~dev_df["DecisionNorm"].isin(ABSENT)].copy()
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
                fig2 = px.line(by_sess, x="Session", y="Abweichung (%)",
                               title="Abweichungsrate pro Session", markers=True)
                fig2.update_layout(xaxis_tickangle=-20)
                st.plotly_chart(fig2, use_container_width=True)

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

            # Deviated votes table
            st.subheader("Abweichende Abstimmungen")
            frac_strength = fraction_majority_strength(combined, members, frac)
            deviated["Fraktionsmehrheit"] = deviated["GroupKey"].map(frac_strength).fillna("—")
            vote_table(deviated, extra_cols=["FracMajority", "Fraktionsmehrheit"])
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
            f"hat keine Mitglieder in der anderen Kammer."
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

    fractions  = sorted(members["FractionAbbr"].dropna().unique())
    frac_labels_list = [frac_label(a) for a in fractions]
    frac_sel   = st.selectbox("Fraktion", frac_labels_list, key="frac_sel")
    frac       = fractions[frac_labels_list.index(frac_sel)]
    frac_name  = members.loc[members["FractionAbbr"] == frac, "FractionName"].iloc[0]

    st.title(frac_label(frac))
    st.caption(frac_name)

    with st.spinner("Lade Daten…"):
        all_votes  = get_all_votes()
        frac_votes = get_fraction_votes(frac)

    if frac_votes.empty:
        st.warning("Keine Daten für diese Fraktion.")
        st.stop()

    # ── Kohäsion ──────────────────────────────────────────────────────────
    st.subheader("Fraktionskohäsion")
    with st.spinner("Berechne Kohäsion…"):
        coh = fraction_cohesion(all_votes, members)

    if not coh.empty:
        this_coh = coh.loc[coh["Fraktion"] == frac, "Kohäsion (%)"]
        c1, c2   = st.columns([1, 3])
        if not this_coh.empty:
            c1.metric(f"Kohäsion {frac_label(frac)}", f"{this_coh.iloc[0]}%")
        coh_plot = coh.copy()
        coh_plot["Fraktion"] = coh_plot["Fraktion"].apply(frac_label)
        fig = px.bar(
            coh_plot, x="Fraktion", y="Kohäsion (%)",
            color="Fraktion", color_discrete_map=frac_color_map(),
            title="Kohäsion aller Fraktionen", hover_data=["Mitglieder"],
            category_orders={"Fraktion": frac_order_labels()},
        )
        fig.update_layout(showlegend=False)
        c2.plotly_chart(fig, use_container_width=True)

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
        fig3.update_layout(height=max(300, len(heatmap_df) * 28))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Kammernvergleich ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Kammernvergleich NR vs SR")
    with st.spinner("Berechne Kammernvergleich…"):
        cc_df = cross_chamber_fraction(frac, all_votes, members)

    if cc_df.empty:
        st.info(
            f"{frac_label(frac)} hat keine Mitglieder in beiden Kammern — "
            "kein Vergleich möglich."
        )
    else:
        n_total  = len(cc_df)
        n_einig  = int(cc_df["Einig"].sum())
        pct_einig = round(n_einig / n_total * 100, 1) if n_total else 0
        kc1, kc2 = st.columns(2)
        kc1.metric("Gemeinsame Geschäfte", n_total)
        kc2.metric("NR & SR einig", f"{pct_einig}%")

        # Show disagreements
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

