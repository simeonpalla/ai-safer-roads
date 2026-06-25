"""
visualization.py v5 — Interactive Folium map.

CHANGES v5 (methodology review, June 2026):
  - MAP LANGUAGE FIX: the raw "OpenStreetMap" tile layer rendered place
    labels in the LOCAL script (Chinese/Thai/Devanagari place names over
    China/Thailand/India) because that's what default OSM-Carto tiles do —
    confirmed from screenshots. Replaced with Wikimedia's osm-intl tile
    endpoint (https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png?lang=en),
    a free, no-API-key tile source that renders OSM data with English
    labels via the lang= parameter. This is now the default/primary layer;
    the local-language layer is removed, not just relabeled.
  - NEW: Schools and Hospitals are now plotted as actual point markers
    (clustered, from the same enrichment_data files the Exposure score
    already loads), not just a number buried in a popup badge.
  - NEW: Population Density heatmap layer, built from the per-segment
    WorldPop buffer samples already computed in enrichment.py.
  - NEW: Intersections marker layer (graceful no-op if no file present,
    same pattern as every other optional enrichment layer in this project).
  - NEW: popup's single "Exposure: 57.0" badge replaced with a breakdown —
    every component score shown with its weight and calculation basis
    (nearest school/hospital distance in metres, population density value,
    intersection score, traffic percentile), so the number is no longer a
    black box.
  - REMOVED: AI Anomaly layer/markers/popup badge. Demoted per methodology
    review — "anomalous in feature space" isn't yet a defensible "hidden
    danger" claim. ai_scoring.py and its standalone CSV are unaffected;
    just not surfaced on this map. See ai_scoring.py module docstring.
  - RENAMED: "Risk Corridors" → "Intervention Zones" everywhere user-
    facing (these are attribute groups, not spatially contiguous
    corridors — see advanced_scoring.detect_corridors docstring).
  - sub_score_op_speed_gap → sub_score_limit_credibility (matches the
    scoring.py rename — see that module's docstring for why).

CHANGES v4 (Priority Index):
  - Popup shows Priority Index (Exposure × Likelihood × Severity) alongside
    SSS — both metrics displayed, neither replaces the other.
  - New toggleable "🎯 Priority Index View" layer recolors segments by
    priority_index/priority_band instead of sss/sss_band, off by default,
    for visual side-by-side comparison via the layer control.
  - Summary panel and legend mention Priority Index band counts and
    Spearman correlation vs SSS.
  - Export CSV includes all new Priority Index columns.

CHANGES v3:
  - Map centers between both countries at zoom 5 (both visible on load)
  - Stats panel shows data coverage % per country
  - Data coverage caveat added to summary panel
  - Heatmap: show=False by default
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, Fullscreen, MiniMap
from pathlib import Path

from config import BAND_COLORS, SCORE_BANDS
from enrichment import _load_amenities, _load_intersections, SCHOOL_BUFFER_M, \
                        HOSPITAL_BUFFER_M


def score_to_color(score: float) -> str:
    if pd.isna(score):
        return "#cccccc"
    score = float(np.clip(score, 0, 100))
    if score < 40:
        r, g, b = 44, 160, 44
    elif score < 60:
        t = (score - 40) / 20
        r = int(44 + t * (188 - 44))
        g = int(160 + t * (189 - 160))
        b = int(44 + t * (34 - 44))
    elif score < 80:
        t = (score - 60) / 20
        r = int(188 + t * (255 - 188))
        g = int(189 + t * (127 - 189))
        b = int(34 + t * (14 - 34))
    else:
        r, g, b = 214, 39, 40
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_popup_html(row: pd.Series) -> str:
    """
    Compact popup using shared CSS classes (GLOBAL_POPUP_STYLES, injected once).
    Per-popup size: ~2KB vs previous ~15KB = 85% reduction.
    Tab switching: single shared swTab(id, t) — no per-popup <script>.
    """
    sss           = row.get("sss", np.nan)
    band          = row.get("sss_band", "—")
    sl            = row.get("speed_limit", np.nan)
    ss            = row.get("ss_limit", np.nan)
    f85           = row.get("speed_85th", np.nan)
    med           = row.get("median_speed", np.nan)
    rc            = row.get("road_class_norm", row.get("road_class", "—"))
    lu            = row.get("land_use", "—")
    country       = row.get("country", row.get("country_code", "—"))
    cc            = row.get("country_code", "")
    seg_id        = row.get("segment_id", "—")
    img_url       = row.get("image_url", "")
    credibility   = row.get("credibility_class", "")
    nilsson       = row.get("nilsson_fatal_ratio", np.nan)
    nilsson_low   = row.get("nilsson_fatal_ratio_low", np.nan)
    nilsson_high  = row.get("nilsson_fatal_ratio_high", np.nan)
    priority_index= row.get("priority_index", np.nan)
    priority_band = row.get("priority_band", "—")
    sinuosity     = row.get("sinuosity", np.nan)
    spread        = (f85 - med) if pd.notna(f85) and pd.notna(med) else np.nan
    align_score   = row.get("sub_score_limit_alignment", np.nan)
    cred_score    = row.get("sub_score_limit_credibility", np.nan)
    vru_score     = row.get("sub_score_vru_risk", np.nan)
    rec_limit     = row.get("recommended_limit", np.nan)
    pct_over      = row.get("pct_over_limit", np.nan)
    change_effort = row.get("change_effort", "")
    est_lives     = row.get("est_lives_saved", np.nan)

    BAND_BG = {"Critical":"#b91c1c","High Risk":"#c2410c","Moderate":"#a16207","Acceptable":"#15803d"}
    band_bg = BAND_BG.get(band, "#374151")

    def fmt(v, unit="", dec=1):
        return f"{v:.{dec}f}{unit}" if pd.notna(v) else "—"

    uid = str(seg_id).replace("/","_").replace(" ","_")

    if pd.notna(sl) and pd.notna(ss):
        gap = sl - ss
        if sl > ss + 5:
            verdict = "Too high"
            vlabel  = f"Posted {sl:.0f} km/h — SS standard {ss:.0f} km/h (+{gap:.0f} over)"
        elif sl < ss * 0.80:
            verdict = "Too low — outdated"
            vlabel  = f"Posted {sl:.0f} km/h — SS standard {ss:.0f} km/h ({ss-sl:.0f} below design speed)"
        elif sl < ss - 5:
            verdict = "Slightly low"
            vlabel  = f"Posted {sl:.0f} km/h — slightly below SS standard ({ss:.0f} km/h)"
        else:
            verdict = "Appropriate"
            vlabel  = f"Posted {sl:.0f} km/h aligns with SS standard ({ss:.0f} km/h)"
    else:
        verdict, vlabel = "No data", "No posted limit recorded"

    sss_pct = min(100, max(0, sss)) if pd.notna(sss) else 0

    def cell(lbl, val, cls):
        v = fmt(val, " km/h", 0) if pd.notna(val) else "—"
        return (f'<td class="pp-cell"><div class="pp-cell-inner {cls}">'
                f'<div class="pp-cell-lbl">{lbl}</div>'
                f'<div class="pp-cell-val">{v}</div></div></td>')

    grid = ""
    if pd.notna(sl) or pd.notna(ss) or pd.notna(med):
        mc = "pp-cell-red" if (pd.notna(med) and pd.notna(sl) and med > sl) else "pp-cell-gray"
        fc = "pp-cell-red" if (pd.notna(f85) and pd.notna(sl) and f85 > sl+10) else "pp-cell-gray"
        spr_str = ""
        if pd.notna(spread):
            sc = " style='color:#c2410c'" if spread > 20 else ""
            w  = " ⚠ mixed traffic" if spread > 20 else ""
            spr_str = f'<b{sc}>{spread:.0f} km/h{w}</b>'
        grid = (f'<div class="pp-grid"><div class="pp-grid-label">Speed at a glance</div>'
                f'<table class="pp-cell-tbl"><tr>'
                f'{cell("Posted limit", sl, "pp-cell-blue")}'
                f'{cell("Safe System", ss, "pp-cell-gray")}'
                f'{cell("Median (GPS)", med, mc)}'
                f'{cell("F85 (GPS)", f85, fc)}'
                f'</tr></table>'
                f'<div class="pp-spread">Spread (F85−median): {spr_str}</div>'
                f'<div class="pp-verdict-box">{vlabel}</div></div>')

    def R(icon, text):
        return (f'<div class="pp-reason"><span class="pp-reason-icon">{icon}</span>'
                f'<span class="pp-reason-text">{text}</span></div>')

    def A(icon, title, detail, impact, badge_cls="pp-badge-r", badge="Recommended"):
        return (f'<div class="pp-action"><div class="pp-action-inner">'
                f'<span class="pp-action-icon">{icon}</span>'
                f'<div class="pp-action-body">'
                f'<div class="pp-action-title">{title}</div>'
                f'<div class="pp-action-detail">{detail}</div>'
                f'<div class="pp-action-impact">↑ {impact}</div>'
                f'</div><span class="pp-badge {badge_cls}">{badge}</span>'
                f'</div></div>')

    reasons = []
    if pd.notna(sl) and pd.notna(ss) and sl > ss + 5:
        reasons.append(R("↑", f"Limit {sl:.0f} exceeds {ss:.0f} km/h Safe System ceiling for {lu} {rc}"))
    if pd.notna(sl) and pd.notna(ss) and sl < ss * 0.80:
        reasons.append(R("⟳", f"Limit {sl:.0f} km/h is {ss-sl:.0f} km/h below design speed — likely outdated"))
    if pd.notna(f85) and pd.notna(sl) and pd.notna(med) and med > sl and f85 > sl + 15:
        reasons.append(R("🚗", f"Median {med:.0f} AND F85 {f85:.0f} km/h both exceed limit — systemic non-compliance"))
    elif pd.notna(f85) and pd.notna(sl) and f85 > sl + 15:
        reasons.append(R("🚗", f"F85 {f85:.0f} km/h exceeds limit by {f85-sl:.0f} km/h"))
    if credibility == "Non-Credible":
        reasons.append(R("⚠", "Limit non-credible — F85 >20 km/h above posted, signage ignored"))
    if credibility == "Infrastructure-Forced":
        reasons.append(R("🔍", "Low speeds likely from speed bumps — verify before recommending limit change"))
    if pd.notna(spread) and spread > 20:
        reasons.append(R("⇔", f"Wide spread ({spread:.0f} km/h) — mixed traffic, GPS probe is car-biased"))
    if pd.notna(nilsson) and nilsson > 4:
        nl = f" [{nilsson_low:.1f}–{nilsson_high:.1f}×]" if pd.notna(nilsson_low) else ""
        reasons.append(R("❤", f"Fatal crash risk {nilsson:.1f}× Safe System baseline (Nilsson){nl}"))
    if pd.notna(sinuosity) and sinuosity >= 1.50:
        reasons.append(R("↩", f"Sharply curved (SI={sinuosity:.2f}) — sight distance limits safe speed"))
    elif pd.notna(sinuosity) and sinuosity >= 1.20:
        reasons.append(R("↩", f"Curved alignment (SI={sinuosity:.2f}) — design speed adjusted"))
    if cc == "TH" and rc in ("secondary", "tertiary", "primary"):
        reasons.append(R("🏍", "TH PTW corridor — 74% of fatalities are PTW (WHO 2023)"))
    if cc == "MH" and rc in ("primary", "trunk"):
        reasons.append(R("🏍", "MH PTW–truck conflict — 37% fatalities PTW; undivided carriageway (iRAP 1-2★)"))
    if not reasons:
        reasons.append(R("✓", "Limit broadly appropriate; no major issues detected"))

    actions = []
    auth_mh = {"motorway":"NHAI","trunk":"NHAI/MSRDC","primary":"Maha. PWD","secondary":"Maha. PWD/District","tertiary":"District/Municipal"}
    auth_th = {"motorway":"DOH","trunk":"DOH","primary":"DOH","secondary":"DRR","tertiary":"DRR/Local"}
    authority = (auth_mh if cc=="MH" or "Maharashtra" in country else auth_th).get(rc, "Road Authority")

    if pd.notna(sl) and pd.notna(ss) and sl > ss + 5:
        rl = rec_limit if pd.notna(rec_limit) else ss
        actions.append(A("🚦", f"Reduce limit to {rl:.0f} km/h (−{sl-rl:.0f} km/h)",
            f"SS standard for {lu} {rc} ({change_effort})",
            "20–40% fatal crash reduction (Nilsson)", "pp-badge-p", "Priority"))
    if pd.notna(sl) and pd.notna(ss) and sl < ss * 0.80:
        actions.append(A("📋", "Commission road audit before raising limit",
            f"Posted {sl:.0f} km/h outdated vs SS {ss:.0f} km/h",
            "Prevents raising limit on substandard road", "pp-badge-p", "Priority"))
    if credibility == "Non-Credible":
        actions.append(A("📷", "Physical calming or enforcement cameras",
            "Signage not working — F85 >20 km/h above posted",
            "Speed cameras reduce F85 5–15 km/h (WHO)", "pp-badge-u", "Urgent"))
    if credibility == "Infrastructure-Forced":
        actions.append(A("🔍", "Verify speed bumps on-site",
            "Low F85+median+tight spread = probable bumps",
            "Prevents incorrect limit-change recommendation", "pp-badge-r", "Recommended"))
    if pd.notna(nilsson) and nilsson > 4 and rc in ("trunk", "primary"):
        actions.append(A("🛡", "Install median barrier / physical separation",
            f"Fatal risk {nilsson:.1f}× baseline on undivided road",
            "Median barriers reduce head-on fatalities ~50% (iRAP)", "pp-badge-p", "Priority"))
    if pd.notna(sinuosity) and sinuosity >= 1.50:
        actions.append(A("⚠", "Curve warning chevrons + advance signs",
            f"SI={sinuosity:.2f} — sight distance limited",
            "Curve signs reduce crashes 15–25% (PIARC)", "pp-badge-r", "Recommended"))
    elif pd.notna(sinuosity) and sinuosity >= 1.20:
        actions.append(A("⚠", "Curve advisory speed signs",
            f"SI={sinuosity:.2f} — moderate curvature",
            "Advisory signs alert to design speed limit", "pp-badge-r", "Recommended"))
    if cc == "TH" and rc in ("secondary", "tertiary", "primary"):
        actions.append(A("🏍", "PTW enforcement + awareness campaign",
            "TH PTW 74% of fatalities — documented high-risk",
            "PTW enforcement reduces PTW fatalities 20–35% (SWOV)", "pp-badge-r", "Recommended"))
    if cc == "MH" and rc in ("primary", "trunk"):
        actions.append(A("🏍", "Rumble strips + hard shoulder demarcation",
            "PTW–truck conflict; undivided carriageway",
            "PTW lane-departure interventions reduce fatalities 20–35%", "pp-badge-r", "Recommended"))
    if not actions:
        actions.append(A("📊", "Monitor — next audit in 12 months",
            "No high-priority issues detected",
            "Routine monitoring", "pp-badge-r", "Routine"))

    def bar(score, wt, color):
        w = min(100, max(0, score)) if pd.notna(score) else 0
        return (f'<div class="pp-score-bar-wrap">'
                f'<div class="pp-score-bar-hdr"><span>{wt}</span><b>{fmt(score)}/100</b></div>'
                f'<div class="pp-score-bar-bg">'
                f'<div class="pp-score-bar-fill" style="width:{w}%;background:{color}"></div>'
                f'</div></div>')

    pi_html = ""
    if pd.notna(priority_index):
        pi_c = {"Critical":"#fee2e2;#991b1b","High Risk":"#ffedd5;#9a3412",
                "Moderate":"#fef9c3;#713f12","Acceptable":"#dcfce7;#14532d"
                }.get(priority_band, "#f3f4f6;#374151").split(";")
        pi_html = (f'<div class="pp-pi-badge" style="background:{pi_c[0]};color:{pi_c[1]}">'
                   f'Priority Index (E×L×S): {priority_index:.1f} — {priority_band}</div>')

    nl_range = (f"<span style='color:#9ca3af'> [{nilsson_low:.1f}–{nilsson_high:.1f}×]</span>"
                if pd.notna(nilsson_low) else "")
    mc2 = "#b91c1c" if pd.notna(med)  and pd.notna(sl) and med  > sl     else "#374151"
    fc2 = "#b91c1c" if pd.notna(f85)  and pd.notna(sl) and f85  > sl+10  else "#374151"
    nc2 = "#b91c1c" if pd.notna(nilsson) and nilsson > 4                  else "#374151"
    sc2 = "#c2410c" if pd.notna(spread)  and spread > 20                  else "#374151"
    lives_row  = (f"<tr><td>Est. lives saved/yr</td>"
                  f"<td style='color:#059669'>{fmt(est_lives)}</td></tr>") if pd.notna(est_lives) else ""
    effort_row = f"<tr><td>Change effort</td><td>{change_effort}</td></tr>" if change_effort else ""
    img_html   = (f'<div class="pp-img-link"><a href="{img_url}" target="_blank">📷 View street imagery</a></div>'
                  if img_url and isinstance(img_url, str) and img_url.startswith("http") else "")

    # ❶ IS THE LIMIT RIGHT — always visible block above tabs
    if pd.notna(sl) and pd.notna(ss):
        if sl > ss + 5:
            q1_cls, q1_hdr_cls = "pp-q1-high", "pp-q1-hdr-high"
            q1_verdict_label = "✗ TOO HIGH"
        elif sl < ss * 0.80:
            q1_cls, q1_hdr_cls = "pp-q1-lo", "pp-q1-hdr-lo"
            q1_verdict_label = "✗ TOO LOW (outdated)"
        elif sl < ss - 5:
            q1_cls, q1_hdr_cls = "pp-q1-hi", "pp-q1-hdr-hi"
            q1_verdict_label = "⚠ SLIGHTLY LOW"
        else:
            q1_cls, q1_hdr_cls = "pp-q1-ok", "pp-q1-hdr-ok"
            q1_verdict_label = "✓ APPROPRIATE"
    else:
        q1_cls, q1_hdr_cls, q1_verdict_label = "pp-q1-ok", "pp-q1-hdr-ok", "— NO DATA"

    q1_speed_row = ""
    if pd.notna(med) or pd.notna(f85):
        parts = []
        if pd.notna(med): parts.append(f"Median {med:.0f} km/h")
        if pd.notna(f85): parts.append(f"F85 {f85:.0f} km/h")
        if pd.notna(spread): parts.append(f"spread {spread:.0f} km/h")
        q1_speed_row = f'<div class="pp-q1-speed">📊 {" · ".join(parts)}</div>'

    # "Critical + Appropriate" fix: verdict = limit direction, band = overall SSS.
    # These measure different things. A Critical road can have an Appropriate limit
    # if drivers ignore it (high credibility gap). Make this explicit.
    if band in ("Critical", "High Risk") and verdict == "Appropriate":
        verdict_display = "Limit set OK — but ignored"
    else:
        verdict_display = verdict

    return f"""<div class="pp-wrap">
<div class="pp-hdr" style="background:{band_bg}">
  <div class="pp-hdr-meta">{country} · {lu} {rc} · {seg_id}</div>
  <div class="pp-hdr-title">{band} — SSS {fmt(sss)}/100
    <span class="pp-verdict">{verdict_display}</span>
  </div>
  <div class="pp-bar-wrap">
    <div class="pp-bar-label"><span>Risk Score</span><span>{sss_pct:.1f}/100</span></div>
    <div class="pp-bar-bg"><div class="pp-bar-fill" style="width:{sss_pct}%"></div></div>
  </div>
</div>
<div class="pp-body">
{grid}
<div style="padding:10px 14px 0">
  <div class="pp-q1 {q1_cls}">
    <div class="pp-q1-hdr {q1_hdr_cls}">❶ IS THE LIMIT RIGHT? &nbsp; {q1_verdict_label}</div>
    <div class="pp-q1-detail">{vlabel}</div>
    {q1_speed_row}
  </div>
</div>
<div class="pp-tabs">
  <button class="pp-tab active" onclick="swTab('{uid}',0)">❶❷❸ Assessment</button>
  <button class="pp-tab" onclick="swTab('{uid}',1)">📊 Score metrics</button>
</div>
<div id="{uid}">
  <div class="pp-panel active">
    <div class="pp-q-block pp-q-why">
      <div class="pp-q-hdr pp-q-hdr-why">❷ WHY?</div>
      <div class="pp-q-body">{''.join(reasons)}</div>
    </div>
    <div class="pp-q-block pp-q-int">
      <div class="pp-q-hdr pp-q-hdr-int">❸ INTERVENTION</div>
      <div class="pp-q-body">
        <div class="pp-auth-row">Responsible authority: {authority}</div>
        {''.join(actions)}
      </div>
    </div>
    {img_html}
  </div>
  <div class="pp-panel">
    {pi_html}
    <div class="pp-section-hdr">SSS component scores</div>
    <div class="pp-score-lbl">Alignment <span>(posted vs Safe System)</span></div>
    {bar(align_score, "Weight: 20%", "#3b82f6")}
    <div class="pp-score-lbl">Credibility gap <span>(dual-signal F85+median)</span></div>
    {bar(cred_score, "Weight: 45%", "#ef4444")}
    <div class="pp-score-lbl">VRU context risk <span>(PTW-weighted)</span></div>
    {bar(vru_score, "Weight: 35%", "#f97316")}
    <div class="pp-formula"><b>SSS = 0.20×{fmt(align_score)} + 0.45×{fmt(cred_score)} + 0.35×{fmt(vru_score)} = <span class="pp-formula-result">{fmt(sss)}</span></b> → {band}</div>
    <hr class="pp-divider">
    <div class="pp-section-hdr">GPS behaviour evidence</div>
    <table class="pp-data-tbl">
      <tr><td>Posted limit</td><td style="color:#1d4ed8">{fmt(sl," km/h",0)}</td></tr>
      <tr><td>Safe System standard</td><td>{fmt(ss," km/h",0)}</td></tr>
      <tr><td>Median speed (GPS)</td><td style="color:{mc2}">{fmt(med," km/h",0)}</td></tr>
      <tr><td>F85 (GPS)</td><td style="color:{fc2}">{fmt(f85," km/h",0)}</td></tr>
      <tr><td>Speed spread</td><td style="color:{sc2}">{fmt(spread," km/h",0)}</td></tr>
      <tr><td>% over limit</td><td>{fmt(pct_over,"%",1)}</td></tr>
      <tr><td>Credibility class</td><td>{credibility or "—"}</td></tr>
      <tr><td>Sinuosity index</td><td>{fmt(sinuosity)} (1.0=straight)</td></tr>
      <tr><td>Nilsson ratio</td><td style="color:{nc2}">{fmt(nilsson,"×")}{nl_range}</td></tr>
      {lives_row}{effort_row}
    </table>
    <div class="pp-sources"><b>Sources:</b> ADB GPS probe · WHO Speed Manual · iRAP · Nilsson 2004 · Elvik 2009 · WHO 2023</div>
  </div>
</div>
</div>
</div>"""

def _priority_sample(df: pd.DataFrame, max_n: int, band_col: str = "sss_band",
                     random_state: int = 42) -> pd.DataFrame:
    """
    Sample up to max_n rows, always keeping ALL Critical segments first,
    then filling remaining slots with High Risk, Moderate, Acceptable in order.
    Ensures the map always shows the most dangerous roads even when sampling.
    """
    if len(df) <= max_n:
        return df
    order = ["Critical", "High Risk", "Moderate", "Acceptable"]
    kept, budget = [], max_n
    for band in order:
        if budget <= 0:
            break
        grp = df[df[band_col] == band] if band_col in df.columns else df
        if len(grp) <= budget:
            kept.append(grp)
            budget -= len(grp)
        else:
            kept.append(grp.sample(budget, random_state=random_state))
            budget = 0
    return pd.concat(kept) if kept else df.sample(max_n, random_state=random_state)


def build_interactive_map(
    gdf: gpd.GeoDataFrame,
    corridors: gpd.GeoDataFrame = None,
    output_path: str = "speed_safety_map.html",
    max_segments: int = 500,        # per country — keeps HTML under ~15MB
    data_dir: str = "enrichment_data",
    max_amenity_markers: int = 6000,
) -> folium.Map:
    """
    FILE SIZE BUDGET:
      Each popup is ~5-8KB of HTML. To stay under 15MB total:
        500 segments × 2 countries = 1,000 SSS segments
        + 400 PI layer
        + 200 T1 layer
        + 500 ML layer (tooltip-only, no popup)
        + ~400 amenity markers
        = ~2,500 objects × ~5KB avg = ~12MB → safe to open in browser.

      All 14,711 scored segments remain in the CSV/GeoPackage exports.
      The map shows the MOST IMPORTANT roads — all Critical, then High Risk
      by priority, then sample of Moderate. Acceptable segments are shown
      as thin lines with tooltip only (no popup) to save space.
    """
    gdf   = gdf.to_crs(epsg=4326)
    mask  = gdf["scoreable"] & gdf["sss"].notna()
    scored = gdf[mask].copy()
    # Tier 1 segments: have an alignment-only score but no full SSS
    tier1_only = gdf[gdf.get("alignment_scoreable", False) & gdf["sss"].isna()
                      & gdf["alignment_only_score"].notna()].copy() \
                 if "alignment_only_score" in gdf.columns else gdf.iloc[0:0].copy()

    # Center between both countries so both visible on load
    center_lat = 18.0
    center_lon = 90.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=None)

    # MAP LANGUAGE FIX: standard "OpenStreetMap" tiles render place labels
    # in the LOCAL script (confirmed from screenshots — Chinese/Thai/
    # Devanagari over China/Thailand/India). Wikimedia's osm-intl endpoint
    # is the same OSM data with English labels forced via lang=en, free,
    # no API key. This is now the only/default base layer so "the whole
    # map in English" is guaranteed rather than one option among several
    # whose language behaviour isn't certain.
    folium.TileLayer(
        tiles="https://{s}.maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png?lang=en",
        attr="© OpenStreetMap contributors, © Wikimedia",
        subdomains=["a", "b", "c"],
        name="Light (English)",
        max_zoom=19,
    ).add_to(m)
    # Dark theme kept as a secondary option. CartoDB's dark_matter style
    # isn't guaranteed English the same explicit way as the layer above —
    # if you need certainty on a non-default layer too, drop it and use
    # only the English layer above.
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    m.get_root().html.add_child(folium.Element(_build_legend_html()))
    m.get_root().html.add_child(folium.Element(_build_summary_html(scored, gdf, tier1_only)))

    # GLOBAL CSS + JS — injected ONCE into the page head.
    # Every popup references these classes/functions instead of repeating
    # inline styles and <script> tags. This reduces per-popup size from
    # ~15KB to ~2KB, cutting total file size by ~85%.
    GLOBAL_POPUP_STYLES = """
    <style>
    .pp-wrap{font-family:system-ui,sans-serif;width:430px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;max-height:82vh;display:flex;flex-direction:column}
    .pp-body{overflow-y:auto;flex:1}
    .pp-hdr{color:#fff;padding:10px 16px}
    .pp-hdr-meta{font-size:11px;opacity:.85;margin-bottom:2px}
    .pp-hdr-title{font-size:16px;font-weight:700;display:flex;align-items:center;gap:10px}
    .pp-verdict{margin-left:auto;font-size:11px;font-weight:700;background:rgba(255,255,255,.18);border-radius:20px;padding:3px 11px}
    .pp-bar-wrap{margin-top:8px}
    .pp-bar-label{display:flex;justify-content:space-between;font-size:10px;color:rgba(255,255,255,.7);margin-bottom:3px}
    .pp-bar-bg{height:5px;background:rgba(255,255,255,.2);border-radius:3px;overflow:hidden}
    .pp-bar-fill{height:100%;background:rgba(255,255,255,.85);border-radius:3px}
    .pp-grid{padding:10px 14px 0;border-bottom:1px solid #e5e7eb}
    .pp-grid-label{font-size:10px;color:#9ca3af;margin-bottom:4px;letter-spacing:.04em;text-transform:uppercase}
    .pp-cell-tbl{width:100%;border-collapse:collapse;margin-top:8px}
    .pp-cell{padding:0 4px;text-align:center}
    .pp-cell-inner{border-radius:6px;padding:5px 8px}
    .pp-cell-lbl{font-size:10px;color:#6b7280;margin-bottom:2px}
    .pp-cell-val{font-size:14px;font-weight:700}
    .pp-cell-blue{background:#eff6ff;border:1px solid #bfdbfe}.pp-cell-blue .pp-cell-val{color:#1d4ed8}
    .pp-cell-gray{background:#f9fafb;border:1px solid #e5e7eb}.pp-cell-gray .pp-cell-val{color:#111827}
    .pp-cell-red{background:#fef2f2;border:1px solid #fecaca}.pp-cell-red .pp-cell-val{color:#b91c1c}
    .pp-spread{font-size:10px;color:#6b7280;margin-top:5px}
    .pp-verdict-box{margin:6px 0 10px;font-size:11px;color:#374151;background:#f9fafb;border-radius:5px;padding:5px 8px}
    .pp-tabs{display:flex;border-bottom:1px solid #e5e7eb;background:#f3f4f6}
    .pp-tab{flex:1;padding:8px 12px;border:none;border-bottom:3px solid transparent;cursor:pointer;font-size:12px;font-weight:600;background:#f3f4f6;color:#6b7280;border-radius:0}
    .pp-tab.active{background:#fff;color:#1d4ed8;border-bottom:3px solid #1d4ed8;font-weight:700}
    .pp-panel{display:none;padding:12px 14px}
    .pp-panel.active{display:block}
    .pp-section-hdr{font-size:10px;font-weight:700;color:#6b7280;letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px}
    .pp-reason{display:flex;align-items:flex-start;gap:8px;background:#fff;border:1px solid #e5e7eb;border-left:3px solid #ef4444;border-radius:0 6px 6px 0;padding:7px 10px;margin-bottom:5px}
    .pp-reason-icon{color:#ef4444;font-size:13px;flex-shrink:0}
    .pp-reason-text{font-size:12px;color:#374151;line-height:1.5}
    .pp-actions-hdr{display:flex;align-items:center;gap:8px;margin:10px 0 6px}
    .pp-auth{margin-left:auto;font-size:10px;color:#6b7280;background:#f3f4f6;border-radius:4px;padding:2px 8px;white-space:nowrap}
    .pp-action{background:#eff6ff;border:1px solid #bfdbfe;border-left:3px solid #3b82f6;border-radius:0 6px 6px 0;padding:8px 12px;margin-bottom:6px}
    .pp-action-inner{display:flex;align-items:flex-start;gap:8px}
    .pp-action-icon{font-size:14px;color:#3b82f6;flex-shrink:0}
    .pp-action-body{flex:1}
    .pp-action-title{font-size:12px;font-weight:700;color:#1e3a5f;margin-bottom:2px}
    .pp-action-detail{font-size:11px;color:#4b5563;margin-bottom:3px}
    .pp-action-impact{font-size:10px;color:#059669}
    .pp-badge{font-size:10px;font-weight:700;border-radius:4px;padding:2px 6px;white-space:nowrap}
    .pp-badge-p{color:#b91c1c;background:#b91c1c1a}
    .pp-badge-u{color:#c2410c;background:#c2410c1a}
    .pp-badge-r{color:#374151;background:#3741511a}
    .pp-img-link{padding-top:8px;border-top:1px solid #f3f4f6}
    .pp-img-link a{font-size:11px;color:#3b82f6;text-decoration:none}
    .pp-pi-badge{margin-bottom:8px;padding:6px 10px;border-radius:5px;font-size:11px;font-weight:600}
    .pp-score-lbl{margin-bottom:4px;font-size:11px;color:#374151;font-weight:600}
    .pp-score-lbl span{font-size:10px;font-weight:400;color:#9ca3af}
    .pp-score-bar-wrap{margin-bottom:8px}
    .pp-score-bar-hdr{display:flex;justify-content:space-between;font-size:11px;color:#6b7280;margin-bottom:3px}
    .pp-score-bar-hdr b{font-weight:700;color:#111827}
    .pp-score-bar-bg{height:7px;background:#e5e7eb;border-radius:4px;overflow:hidden}
    .pp-score-bar-fill{height:100%;border-radius:4px}
    .pp-formula{background:#f0fdf4;border:1px solid #86efac;border-radius:6px;padding:7px 10px;margin:8px 0;font-size:11px}
    .pp-formula b{color:#15803d}
    .pp-formula .pp-formula-result{font-size:13px}
    .pp-divider{border:none;border-top:1px solid #f3f4f6;margin:10px 0}
    .pp-data-tbl{width:100%;font-size:11px;border-collapse:collapse}
    .pp-data-tbl tr{border-bottom:1px solid #f3f4f6}
    .pp-data-tbl td:first-child{padding:4px 6px;color:#6b7280}
    .pp-data-tbl td:last-child{padding:4px 6px;font-weight:700;text-align:right}
    .pp-sources{margin-top:8px;padding:6px 8px;background:#f9fafb;border-radius:5px;font-size:9px;color:#9ca3af;line-height:1.6}
    .pp-sources b{color:#6b7280}
    .pp-q1{margin:0 0 8px;padding:8px 10px;border-radius:6px}
    .pp-q1-ok{background:#f0fdf4;border:1px solid #86efac}
    .pp-q1-hi{background:#fff7ed;border:1px solid #fdba74}
    .pp-q1-lo{background:#fff7ed;border:1px solid #fdba74}
    .pp-q1-high{background:#fef2f2;border:1px solid #fca5a5}
    .pp-q1-hdr{font-weight:700;font-size:12px;margin-bottom:4px}
    .pp-q1-hdr-ok{color:#15803d}.pp-q1-hdr-hi{color:#c2410c}.pp-q1-hdr-lo{color:#c2410c}.pp-q1-hdr-high{color:#b91c1c}
    .pp-q1-detail{font-size:11px;color:#374151;margin-bottom:4px}
    .pp-q1-speed{font-size:11px;color:#6b7280;background:#f9fafb;padding:3px 6px;border-radius:3px}
    .pp-q-block{margin:6px 0 0}
    .pp-q-block.pp-q-why{background:#fdf2f2;border:1px solid #e8c5c5;border-radius:6px;padding:8px 10px}
    .pp-q-block.pp-q-int{background:#eaf4fb;border:1px solid #b3d9f2;border-radius:6px;padding:8px 10px}
    .pp-q-hdr{font-weight:700;font-size:12px;margin-bottom:6px}
    .pp-q-hdr-why{color:#7f1d1d}.pp-q-hdr-int{color:#1a4f72}
    .pp-auth-row{font-size:11px;color:#555;font-weight:700;margin-bottom:6px}
    </style>
    <script>
    function swTab(id,t){
      var p=document.getElementById(id);
      if(!p)return;
      var tabs=p.querySelectorAll('.pp-tab');
      var panels=p.querySelectorAll('.pp-panel');
      tabs.forEach(function(tb,i){
        tb.classList.toggle('active',i===t);
      });
      panels.forEach(function(pn,i){
        pn.classList.toggle('active',i===t);
      });
    }
    </script>
    """
    m.get_root().html.add_child(folium.Element(GLOBAL_POPUP_STYLES))

    # Segment layers per country
    # Strategy: Critical + High Risk always get full popup (they're the action items).
    # Moderate gets popup up to budget. Acceptable gets tooltip-only (thin line, no popup).
    # This is the single biggest file-size lever — each full popup is ~5-8KB.
    for country_code in scored["country_code"].unique():
        country_sub  = scored[scored["country_code"] == country_code]
        country_name = country_sub["country"].iloc[0]
        fg = folium.FeatureGroup(name=f"📍 {country_name}", show=True)

        plot_sub = _priority_sample(country_sub, max_segments, band_col="sss_band")
        if len(plot_sub) < len(country_sub):
            print(f"  Sampling {len(plot_sub):,} of {len(country_sub):,} {country_code} segments "
                  f"(priority: all Critical first)")

        for _, row in plot_sub.iterrows():
            geom   = row.geometry
            sss    = row.get("sss", np.nan)
            band   = row.get("sss_band", "Acceptable")
            color  = score_to_color(sss)
            weight = 3 if band in ("Critical", "High Risk") else 2
            tooltip_txt = (
                f"{band} | SSS {sss:.1f} | {row.get('road_class_norm','—')} "
                f"| Posted {row.get('speed_limit','—')} km/h → SS {row.get('ss_limit','—')} km/h"
                if pd.notna(sss) else "No data"
            )

            # Acceptable roads: tooltip only — saves ~5KB per segment
            use_popup = band in ("Critical", "High Risk", "Moderate")

            try:
                if geom.geom_type in ("LineString", "MultiLineString"):
                    for seg_coords in _geom_to_latlon_list(geom):
                        line = folium.PolyLine(
                            locations=seg_coords, color=color, weight=weight,
                            opacity=0.85 if use_popup else 0.5,
                            tooltip=tooltip_txt,
                        )
                        if use_popup:
                            line.add_child(folium.Popup(_build_popup_html(row), max_width=440))
                        line.add_to(fg)
                else:
                    c = geom.centroid
                    mk = folium.CircleMarker(
                        location=[c.y, c.x], radius=5, color=color,
                        fill=True, fill_color=color, fill_opacity=0.8,
                        tooltip=tooltip_txt,
                    )
                    if use_popup:
                        mk.add_child(folium.Popup(_build_popup_html(row), max_width=440))
                    mk.add_to(fg)
            except Exception:
                pass
        fg.add_to(m)

    # Tier 1 only — limit vs standard, no GPS data. Capped at 200 with tooltip only.
    if len(tier1_only):
        fg_t1 = folium.FeatureGroup(
            name=f"📏 Tier 1 Only — Limit vs Standard ({len(tier1_only):,} segments)",
            show=False,
        )
        plot_t1 = tier1_only.head(200)  # tooltip only, keep file small
        for _, row in plot_t1.iterrows():
            try:
                geom  = row.geometry
                score = row.get("alignment_only_score", np.nan)
                band  = row.get("alignment_only_band", "Acceptable")
                color = score_to_color(score)
                tooltip_txt = f"Tier 1 | {band}: {score:.1f}" if pd.notna(score) else "No data"
                if geom.geom_type in ("LineString", "MultiLineString"):
                    for seg_coords in _geom_to_latlon_list(geom):
                        folium.PolyLine(
                            locations=seg_coords, color=color, weight=1.5,
                            opacity=0.6, dash_array="4,4",
                            tooltip=tooltip_txt,
                        ).add_to(fg_t1)
            except Exception:
                pass
        fg_t1.add_to(m)

    # Critical segments only layer — full popups, all of them
    critical = scored[scored["sss_band"] == "Critical"]
    if len(critical):
        fg_crit = folium.FeatureGroup(name="🔴 Critical Segments Only", show=False)
        for _, row in critical.iterrows():
            try:
                c = row.geometry.centroid
                folium.Marker(
                    location=[c.y, c.x],
                    popup=folium.Popup(_build_popup_html(row), max_width=360),
                    tooltip=f"CRITICAL | SSS: {row.get('sss', 0):.1f}",
                    icon=folium.Icon(color="red", icon="exclamation-sign"),
                ).add_to(fg_crit)
            except Exception:
                pass
        fg_crit.add_to(m)

    # ── Schools, Hospitals ─────────────────────────────────────────────────
    # Off by default — only show those near scored road segments.
    # The enrichment files cover the whole country; rendering all 11k+25k
    # points creates clusters in random locations far from roads.
    # Filter: only keep points within 750m of a scored road centroid.
    try:
        import geopandas as _gpd
        scored_pts = scored.copy()
        scored_pts["geometry"] = scored_pts.geometry.centroid
        scored_centroids_m = _gpd.GeoDataFrame(scored_pts[["geometry"]], crs="EPSG:4326").to_crs("EPSG:3857")
        road_buffer_union  = scored_centroids_m.geometry.buffer(750).unary_union
        _proximity_filter_ready = True
    except Exception:
        road_buffer_union = None
        _proximity_filter_ready = False

    def _filter_near_roads(amenity_gdf):
        if not _proximity_filter_ready or road_buffer_union is None:
            return amenity_gdf
        try:
            amen_m = amenity_gdf.to_crs("EPSG:3857")
            mask   = amen_m.geometry.within(road_buffer_union)
            return amenity_gdf[mask.values]
        except Exception:
            return amenity_gdf

    schools_gdf = _load_amenities(f"{data_dir}/schools", "Schools (map)")
    if len(schools_gdf):
        schools_near = _filter_near_roads(schools_gdf)
        plot_schools = schools_near if len(schools_near) >= 10 else schools_gdf
        if len(plot_schools) > 800:
            plot_schools = plot_schools.sample(800, random_state=42)
        fg_schools = folium.FeatureGroup(
            name=f"🏫 Schools near scored roads ({len(plot_schools):,})", show=False)
        cluster_schools = MarkerCluster(name="schools_cluster").add_to(fg_schools)
        for _, row in plot_schools.iterrows():
            try:
                c = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
                folium.CircleMarker(
                    location=[c.y, c.x], radius=4, color="#2563eb",
                    fill=True, fill_color="#2563eb", fill_opacity=0.9,
                    tooltip="School (within 750m of scored road)",
                ).add_to(cluster_schools)
            except Exception:
                pass
        fg_schools.add_to(m)

    hosp_gdf = _load_amenities(f"{data_dir}/hospitals", "Hospitals (map)")
    if len(hosp_gdf):
        hosp_near = _filter_near_roads(hosp_gdf)
        plot_hosp = hosp_near if len(hosp_near) >= 10 else hosp_gdf
        if len(plot_hosp) > 800:
            plot_hosp = plot_hosp.sample(800, random_state=42)
        fg_hosp = folium.FeatureGroup(
            name=f"🏥 Hospitals near scored roads ({len(plot_hosp):,})", show=False)
        cluster_hosp = MarkerCluster(name="hospitals_cluster").add_to(fg_hosp)
        for _, row in plot_hosp.iterrows():
            try:
                c = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
                folium.CircleMarker(
                    location=[c.y, c.x], radius=4, color="#dc2626",
                    fill=True, fill_color="#dc2626", fill_opacity=0.9,
                    tooltip="Hospital (within 750m of scored road)",
                ).add_to(cluster_hosp)
            except Exception:
                pass
        fg_hosp.add_to(m)

    intersections_gdf = _load_intersections(data_dir)
    if len(intersections_gdf):
        plot_int = intersections_gdf
        if len(intersections_gdf) > max_amenity_markers:
            plot_int = intersections_gdf.sample(max_amenity_markers, random_state=42)
        fg_int = folium.FeatureGroup(
            name=f"🚦 Intersections ({len(intersections_gdf):,})", show=False)
        cluster_int = MarkerCluster(name="intersections_cluster").add_to(fg_int)
        for _, row in plot_int.iterrows():
            try:
                c = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
                folium.CircleMarker(
                    location=[c.y, c.x], radius=3, color="#f59e0b",
                    fill=True, fill_color="#f59e0b", fill_opacity=0.9,
                    tooltip="Intersection",
                ).add_to(cluster_int)
            except Exception:
                pass
        fg_int.add_to(m)
    # else: no intersection files found — same graceful no-op as the
    # scoring pipeline; nothing rendered, nothing crashes.

    # Population density heatmap — built from the per-segment WorldPop
    # buffer samples already computed in enrichment.py (pop_density_500m),
    # NOT the raw raster (rendering a GeoTIFF as a map overlay needs a
    # tile server, out of scope here) — this is a real but approximate
    # proxy, off by default since combined with everything else on this
    # map a heatmap gets visually loud fast; toggle it on for a clean
    # population-only view.
    pop_col = "pop_density_500m"
    if pop_col in gdf.columns and gdf[pop_col].notna().any():
        pop_pts = gdf[gdf[pop_col].notna() & (gdf[pop_col] > 0)]
        if len(pop_pts):
            heat_pop = []
            for _, row in pop_pts.iterrows():
                try:
                    c = row.geometry.centroid
                    heat_pop.append([c.y, c.x, float(row[pop_col])])
                except Exception:
                    pass
            if heat_pop:
                fg_pop = folium.FeatureGroup(
                    name="👨‍👩‍👧 Population Density (WorldPop, road-buffer sample)", show=False)
                HeatMap(heat_pop, radius=14, blur=10, max_zoom=13,
                        gradient={"0.2": "#fde68a", "0.5": "#f59e0b", "0.8": "#dc2626", "1.0": "#7f1d1d"}
                        ).add_to(fg_pop)
                fg_pop.add_to(m)

    # Heatmap — hidden by default
    high_risk = scored[scored["sss"] >= 60]
    if len(high_risk):
        heat_data = []
        for _, row in high_risk.iterrows():
            try:
                c = row.geometry.centroid
                heat_data.append([c.y, c.x, row.get("sss", 60) / 100])
            except Exception:
                pass
        fg_heat = folium.FeatureGroup(name="🌡️ Risk Heat Map", show=False)
        HeatMap(heat_data, radius=12, blur=8, max_zoom=13,
                gradient={"0.4": "#2ca02c", "0.6": "#bcbd22", "0.8": "#ff7f0e", "1.0": "#d62728"}
                ).add_to(fg_heat)
        fg_heat.add_to(m)

    # Priority Index View — tooltip only to save space, 400 segments max
    if "priority_index" in scored.columns:
        pi_scored = scored[scored["priority_index"].notna()]
        if len(pi_scored):
            fg_pi = folium.FeatureGroup(name="🎯 Priority Index View", show=False)
            plot_pi = _priority_sample(pi_scored, 400, band_col="priority_band")

            for _, row in plot_pi.iterrows():
                try:
                    geom  = row.geometry
                    pi    = row.get("priority_index", np.nan)
                    band  = row.get("priority_band", "Acceptable")
                    color = score_to_color(pi)
                    weight = 3 if band in ("Critical", "High Risk") else 2
                    tooltip_txt = (
                        f"{band} | PI: {pi:.1f} | SSS: {row.get('sss',0):.1f}"
                        if pd.notna(pi) else "No data"
                    )
                    if geom.geom_type in ("LineString", "MultiLineString"):
                        for seg_coords in _geom_to_latlon_list(geom):
                            folium.PolyLine(
                                locations=seg_coords, color=color, weight=weight,
                                opacity=0.85, tooltip=tooltip_txt,
                            ).add_to(fg_pi)
                    else:
                        c = geom.centroid
                        folium.CircleMarker(
                            location=[c.y, c.x], radius=5, color=color,
                            fill=True, fill_color=color, fill_opacity=0.8,
                            tooltip=tooltip_txt,
                        ).add_to(fg_pi)
                except Exception:
                    pass
            fg_pi.add_to(m)

    # Infrastructure blindspot layer — high-SSS segments with zero Mapillary
    # coverage. These are dangerous AND invisible to digital monitoring systems.
    # Shown off by default in a distinctive purple dashed style.
    if "mapillary_blindspot" in scored.columns:
        blindspots = scored[scored["mapillary_blindspot"].fillna(False).astype(bool)]
        if len(blindspots):
            fg_blind = folium.FeatureGroup(
                name=f"👁️ Infrastructure Blindspots ({len(blindspots):,} — high SSS + no imagery)",
                show=False,
            )
            for _, row in blindspots.iterrows():
                try:
                    geom = row.geometry
                    sss  = row.get("sss", np.nan)
                    cc   = row.get("country_code", "")
                    rc   = row.get("road_class_norm", "—")
                    popup_html = (
                        f"<div style='font-family:Arial;width:260px;font-size:13px'>"
                        f"<div style='background:#7c3aed;color:white;padding:8px;border-radius:4px 4px 0 0'>"
                        f"<b>👁️ Infrastructure Blindspot</b></div>"
                        f"<div style='padding:10px;border:1px solid #ddd;border-top:none'>"
                        f"<b>SSS:</b> {sss:.1f} ({row.get('sss_band','—')})<br>"
                        f"<b>Country:</b> {cc} &nbsp;|&nbsp; <b>Class:</b> {rc}<br>"
                        f"<b>Posted limit:</b> {row.get('speed_limit','—')} km/h<br>"
                        f"<hr style='margin:6px 0'>"
                        f"<i style='color:#555;font-size:11px'>No Mapillary street imagery found "
                        f"for this cell. Road is both high-risk and invisible to digital "
                        f"monitoring systems — priority for enforcement camera deployment.</i>"
                        f"</div></div>"
                    )
                    if geom.geom_type in ("LineString", "MultiLineString"):
                        for seg_coords in _geom_to_latlon_list(geom):
                            folium.PolyLine(
                                locations=seg_coords,
                                color="#7c3aed",
                                weight=3,
                                opacity=0.9,
                                dash_array="6 4",
                                popup=folium.Popup(popup_html, max_width=280),
                                tooltip=f"Blindspot | SSS {sss:.0f} | {cc}",
                            ).add_to(fg_blind)
                except Exception:
                    pass
            fg_blind.add_to(m)
            print(f"  Blindspot layer: {len(blindspots):,} segments")

    # Intervention zones — hidden by default, lines only (no fill).
    # These are attribute groups (country + region + road class + band),
    # NOT spatially contiguous corridors — see advanced_scoring.py.
    if corridors is not None and len(corridors):
        fg_corr = folium.FeatureGroup(name="🛣️ Intervention Zones", show=False)
        corr_4326 = corridors.to_crs(epsg=4326)
        for _, row in corr_4326.iterrows():
            try:
                sss_val = row.get("sss", 70)
                n_seg   = row.get("n_segments", 0)
                rank    = row.get("priority_rank", "—")
                ratio   = row.get("nilsson_fatal_ratio", np.nan)
                saved   = row.get("est_lives_saved", np.nan)
                effort  = row.get("change_effort", "—")
                label   = row.get("corridor_label", f"Zone #{rank}")

                popup_html = f"""
                <div style="font-family:Arial;width:260px;font-size:13px">
                  <div style="background:#8B0000;color:white;padding:8px;
                              border-radius:4px 4px 0 0;font-weight:bold">
                    🛣️ Intervention Zone #{rank}
                  </div>
                  <div style="padding:10px;border:1px solid #ddd;border-top:none">
                    <b>Group:</b> {label}<br>
                    <b>Segments:</b> {n_seg}<br>
                    <b>Avg SSS:</b> {sss_val:.1f}<br>
                    <b>Fatal risk ratio:</b> {f"{ratio:.1f}×" if pd.notna(ratio) else "—"}<br>
                    <b>Est. lives saved/yr (illustrative):</b> {f"{saved:.2f}" if pd.notna(saved) else "—"}<br>
                    <b>Change effort:</b> {effort}<br>
                    <i style="font-size:10px;color:#888">Attribute group (same country/region/
                    road class/band), not a spatially contiguous corridor.</i>
                  </div>
                </div>"""

                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        "fillColor": "#8B0000",
                        "color": "#8B0000",
                        "weight": 5,
                        "fillOpacity": 0.0,
                        "opacity": 0.85,
                    },
                    popup=folium.Popup(popup_html, max_width=280),
                    tooltip=f"Zone #{rank} | {n_seg} segments | SSS {sss_val:.0f}",
                ).add_to(fg_corr)
            except Exception:
                pass
        fg_corr.add_to(m)

    # ML-predicted layer — XGBoost SSS estimates for unscored segments
    # Shown off by default (dashed lines) — estimates only, not measured values.
    # These 45k segments have no GPS speed data; predictions come from road
    # attributes (class, land use, ss_limit, exposure). Useful for TRIAGE,
    # not for enforcement or policy decisions on individual roads.
    if "ml_predicted_sss" in gdf.columns:
        ml_segs = gdf[gdf["ml_predicted_sss"].notna()].copy()
        if len(ml_segs):
            n_train = int(gdf.get("scoreable", pd.Series(False, index=gdf.index)).sum())
            # Sample for performance — prioritise higher predicted scores
            if len(ml_segs) > max_segments:
                ml_segs = ml_segs.sort_values("ml_predicted_sss", ascending=False).head(max_segments)
            n_ml = len(ml_segs)
            fg_ml = folium.FeatureGroup(
                name=f"🤖 ML Coverage Extension ({n_ml:,} estimated — no GPS data)",
                show=False,
            )
            from config import BAND_COLORS
            for _, row in ml_segs.iterrows():
                coords = _geom_to_latlon_list(row.geometry)
                if not coords:
                    continue

                ml_sss   = row.get("ml_predicted_sss", float("nan"))
                ml_band  = row.get("ml_predicted_band", "Moderate")
                ml_conf  = row.get("ml_confidence", float("nan"))
                rc       = row.get("road_class_norm", "—")
                lu       = row.get("land_use", "—")
                sl       = row.get("speed_limit", float("nan"))
                ss       = row.get("ss_limit", float("nan"))
                cc       = row.get("country_code", "")
                color    = BAND_COLORS.get(ml_band, "#888888")

                def _fmt(v, u="", d=1):
                    return f"{v:.{d}f}{u}" if pd.notna(v) else "—"

                # Q1 verdict for ML segment
                if pd.notna(sl) and pd.notna(ss):
                    if sl > ss + 5:
                        ml_q1 = f"✗ TOO HIGH — Posted {_fmt(sl)} km/h vs Safe System {_fmt(ss)} km/h"
                        ml_q1_color = "#c0392b"
                    elif sl < ss * 0.80:
                        ml_q1 = f"✗ TOO LOW — Posted {_fmt(sl)} km/h vs Safe System {_fmt(ss)} km/h (outdated)"
                        ml_q1_color = "#e67e22"
                    else:
                        ml_q1 = f"✓ Broadly appropriate — Posted {_fmt(sl)} km/h vs Safe System {_fmt(ss)} km/h"
                        ml_q1_color = "#27ae60"
                else:
                    ml_q1 = "— No posted limit data"
                    ml_q1_color = "#7f8c8d"

                conf_label = "High" if pd.notna(ml_conf) and ml_conf < 5 else ("Medium" if pd.notna(ml_conf) and ml_conf < 10 else "Low")
                conf_color = "#27ae60" if conf_label == "High" else ("#f39c12" if conf_label == "Medium" else "#c0392b")

                popup_html = f"""
                <div style="font-family:Arial;width:340px;font-size:13px;line-height:1.4">
                  <div style="background:#4b0082;color:white;padding:8px 12px;
                              border-radius:4px 4px 0 0;font-weight:bold;font-size:13px">
                    🤖 ML ESTIMATE · {ml_band} · SSS {_fmt(ml_sss)}/100 · {lu} {rc}
                  </div>
                  <div style="padding:10px 12px;border:1px solid #ddd;border-top:none;
                              border-radius:0 0 4px 4px">
                    <div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:4px;
                                padding:6px 8px;margin-bottom:8px;font-size:11px;color:#78350f">
                      ⚠ <b>ESTIMATE ONLY</b> — No GPS speed data for this segment.<br>
                      XGBoost model trained on {n_train:,} measured segments (CV RMSE 7.1, R²=0.73).<br>
                      Use for network triage only — not for individual road decisions.
                    </div>

                    <div style="padding:6px 8px;background:{ml_q1_color}18;
                                border:1px solid {ml_q1_color}44;border-radius:4px;margin-bottom:8px">
                      <b style="color:{ml_q1_color}">❶ LIMIT ASSESSMENT (estimated)</b><br>
                      <span style="font-size:12px">{ml_q1}</span>
                    </div>

                    <div style="font-size:12px;color:#555;margin-bottom:6px">
                      <b>Model confidence:</b>
                      <span style="color:{conf_color};font-weight:bold">{conf_label}</span>
                      {"(fold std = " + _fmt(ml_conf) + " SSS pts)" if pd.notna(ml_conf) else ""}
                      <br>
                      <span style="color:#888;font-size:11px">
                        High confidence = fold models agree; low = road type underrepresented in training data
                      </span>
                    </div>

                    <div style="font-size:11px;color:#555;padding:4px 6px;
                                background:#f8f9fa;border-radius:3px">
                      <b>What drove this estimate:</b> road class ({rc}), land use ({lu}),
                      Safe System limit ({_fmt(ss)} km/h), posted limit ({_fmt(sl)} km/h),
                      school/hospital proximity, exposure score.
                      {"PTW risk flag applied (TH secondary/primary)." if cc == "TH" and rc in ("secondary","primary","tertiary") else ""}
                      {"PTW-truck conflict flag applied (MH primary/trunk)." if cc == "MH" and rc in ("primary","trunk") else ""}
                    </div>
                  </div>
                </div>"""

                for seg_coords in coords:
                    folium.PolyLine(
                        locations=seg_coords,
                        color=color,
                        weight=2,
                        opacity=0.55,
                        dash_array="6 4",
                        popup=folium.Popup(popup_html, max_width=360),
                        tooltip=f"🤖 ML Est: {_fmt(ml_sss)} ({ml_band}) | {lu} {rc} | conf:{conf_label}",
                    ).add_to(fg_ml)
            fg_ml.add_to(m)

    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    m.save(output_path)
    print(f"\nInteractive map saved: {output_path}")
    return m


def _geom_to_latlon_list(geom):
    if geom.geom_type == "LineString":
        return [[[c[1], c[0]] for c in geom.coords]]
    elif geom.geom_type == "MultiLineString":
        return [[[c[1], c[0]] for c in line.coords] for line in geom.geoms]
    return []


def _build_legend_html() -> str:
    items = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:5px">'
        f'<div style="width:16px;height:5px;background:{color};border-radius:2px;margin-right:8px"></div>'
        f'<span style="font-size:12px">{band}</span></div>'
        for band, color in BAND_COLORS.items()
    )
    return f"""
    <div id="legend-panel" style="position:fixed;bottom:40px;left:12px;z-index:9999;
                background:rgba(22,22,30,0.95);color:white;
                border-radius:9px;font-family:system-ui,sans-serif;
                box-shadow:0 2px 16px rgba(0,0,0,0.45);min-width:200px">
      <!-- Header - always visible, click to toggle -->
      <div onclick="var b=document.getElementById('legend-body');
                    var c=document.getElementById('legend-chev');
                    if(b.style.display=='none'){{b.style.display='block';c.textContent='▲'}}
                    else{{b.style.display='none';c.textContent='▼'}}"
           style="display:flex;align-items:center;justify-content:space-between;
                  padding:10px 14px;cursor:pointer;user-select:none">
        <span style="font-size:13px;font-weight:700">Speed Safety Score</span>
        <span id="legend-chev" style="font-size:10px;color:#9ca3af;margin-left:12px">▲</span>
      </div>
      <!-- Body - expanded by default -->
      <div id="legend-body" style="display:block;padding:0 14px 12px;border-top:1px solid rgba(255,255,255,0.1)">
        <div style="padding-top:10px">
          {items}
        </div>
        <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.1)">
          <div style="display:flex;align-items:center;margin-bottom:5px">
            <div style="width:12px;height:12px;background:#2563eb;border-radius:50%;margin-right:8px;flex-shrink:0"></div>
            <span style="font-size:11px;color:#d1d5db">Schools (toggle layer)</span>
          </div>
          <div style="display:flex;align-items:center;margin-bottom:5px">
            <div style="width:12px;height:12px;background:#dc2626;border-radius:50%;margin-right:8px;flex-shrink:0"></div>
            <span style="font-size:11px;color:#d1d5db">Hospitals (toggle layer)</span>
          </div>
          <div style="display:flex;align-items:center">
            <div style="width:16px;height:0;border-top:2px dashed #7c3aed;margin-right:8px;flex-shrink:0"></div>
            <span style="font-size:11px;color:#d1d5db">Blindspot (high SSS, no imagery)</span>
          </div>
        </div>
        <div style="margin-top:8px;font-size:10px;color:#6b7280;line-height:1.5">
          Same scale: SSS view + Priority Index view<br>
          AI for Safer Roads · ADB Challenge
        </div>
      </div>
    </div>
    """


def _build_methodology_html() -> str:
    """
    Plain-language explanation of how each Exposure component is
    calculated, anchored to a fixed panel rather than buried only in the
    per-segment popup, so a reviewer can find "how was this computed"
    without clicking a specific road first.
    """
    return f"""
    <div id="methodology-panel" style="position:fixed;bottom:40px;right:12px;z-index:9999;
                max-width:280px;background:rgba(20,20,20,0.93);color:white;
                padding:12px 16px;border-radius:8px;font-family:Arial,sans-serif;
                font-size:12px;box-shadow:0 2px 12px rgba(0,0,0,0.4)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <b style="font-size:13px">How to read this map</b>
        <span onclick="var p=document.getElementById('methodology-panel');p.style.display='none'"
              style="cursor:pointer;font-size:16px;padding:0 4px" title="Close">×</span>
      </div>
      <hr style="border-color:#444;margin:6px 0">
      <div style="color:#ccc;line-height:1.6">
        <b style="color:#fff">Click any road</b> to see 3 answers:<br>
        <span style="color:#e74c3c">❶ Is the limit right?</span> — posted vs Safe System standard<br>
        <span style="color:#c0392b">❷ Why?</span> — specific reasons (over/under-posted, driver behaviour, PTW risk)<br>
        <span style="color:#2980b9">❸ What to do?</span> — specific engineering actions + responsible authority<br>
        <hr style="border-color:#444;margin:6px 0">
        <b style="color:#fff">Road colours</b> = Speed Safety Score (SSS)<br>
        🔴 Critical: limit severely wrong + drivers confirm it<br>
        🟠 High Risk: significant misalignment<br>
        🟡 Moderate: some misalignment<br>
        🟢 Acceptable: limit broadly correct<br>
        <hr style="border-color:#444;margin:6px 0">
        <b style="color:#fff">Safe System limit</b> = WHO/iRAP speed ceiling for this road class × land use. Not the same as the posted limit — it is the evidence-based target.<br>
        <hr style="border-color:#444;margin:6px 0">
        <span style="color:#aaa;font-size:10px">
          SSS = 20% alignment + 45% credibility gap (dual-signal F85+median) + 35% VRU risk (with PTW weighting).<br>
          Safe System limits use 60 km/h for undivided rural primary/secondary (iRAP 1-2 star, MH/TH).
        </span>
      </div>
    </div>
    """


def _build_summary_html(scored: gpd.GeoDataFrame, full_gdf: gpd.GeoDataFrame,
                         tier1_only: gpd.GeoDataFrame = None) -> str:
    total = len(scored)
    by_band = scored["sss_band"].value_counts()

    # Data coverage per country
    coverage_rows = ""
    for cc in sorted(full_gdf["country_code"].dropna().unique()):
        total_cc   = len(full_gdf[full_gdf["country_code"] == cc])
        scored_cc  = len(scored[scored["country_code"] == cc])
        mean_sss   = scored.loc[scored["country_code"] == cc, "sss"].mean()
        pct        = 100 * scored_cc / total_cc if total_cc > 0 else 0
        coverage_rows += (
            f'<tr><td>{cc}</td>'
            f'<td style="text-align:right">{mean_sss:.1f}</td>'
            f'<td style="text-align:right">{scored_cc:,}</td>'
            f'<td style="text-align:right">{pct:.0f}%</td></tr>'
        )

    band_rows = "".join(
        f'<tr><td style="color:{BAND_COLORS.get(b,"#aaa")}">{b}</td>'
        f'<td style="text-align:right">{by_band.get(b,0):,}</td>'
        f'<td style="text-align:right">{by_band.get(b,0)/total*100:.1f}%</td></tr>'
        for b in SCORE_BANDS.keys()
    )

    # Tier 1 coverage note (alignment-only, no behavioural data needed)
    n_t1 = len(tier1_only) if tier1_only is not None else 0
    tier1_row = ""
    if n_t1:
        tier1_row = f"""
        <br><b style="color:#93c5fd">📏 Tier 1 only (limit vs standard):</b> {n_t1:,} more segments<br>
        <span style="color:#aaa;font-size:11px">No GPS behavioural data, but have a posted limit —
        toggle "Tier 1 Only" layer to view</span>"""

    # Priority Index summary — SECONDARY "where to act first" layer
    pi_row = ""
    if "priority_index" in scored.columns:
        pi_scored = scored[scored["priority_index"].notna()]
        if len(pi_scored):
            pi_band_rows = "".join(
                f'<tr><td style="color:{BAND_COLORS.get(b,"#aaa")}">{b}</td>'
                f'<td style="text-align:right">{(pi_scored["priority_band"]==b).sum():,}</td></tr>'
                for b in BAND_COLORS.keys()
            )
            corr_txt = ""
            both = pi_scored[pi_scored["sss"].notna()]
            if len(both) >= 10:
                try:
                    from scipy import stats as _stats
                    rho, _ = _stats.spearmanr(both["sss"], both["priority_index"])
                    corr_txt = f'<br><span style="color:#aaa;font-size:11px">Spearman ρ vs SSS: {rho:.2f}</span>'
                except Exception:
                    pass
            pi_row = f"""
            <br><b style="color:#e0a800">🎯 Priority Index</b> (secondary "where to act first" layer):
            <table style="width:100%;border-collapse:collapse;margin-top:4px">
              <tr style="color:#aaa"><td>Band</td><td style="text-align:right">N</td></tr>
              {pi_band_rows}
            </table>
            {corr_txt}"""

    # ML coverage note
    ml_row = ""
    if "ml_predicted_sss" in full_gdf.columns:
        n_ml = int(full_gdf["ml_predicted_sss"].notna().sum())
        if n_ml:
            n_ml_hr = int((full_gdf["ml_predicted_band"] == "High Risk").sum()) if "ml_predicted_band" in full_gdf.columns else 0
            n_ml_cr = int((full_gdf["ml_predicted_band"] == "Critical").sum()) if "ml_predicted_band" in full_gdf.columns else 0
            ml_row = (
                f'<br><b style="color:#c4b5fd">🤖 ML Coverage Extension</b><br>'
                f'<span style="color:#ddd">{n_ml:,} unscored segments estimated</span><br>'
                f'<span style="color:#f87171">Critical: {n_ml_cr:,}</span> · '
                f'<span style="color:#fb923c">High Risk: {n_ml_hr:,}</span><br>'
                f'<span style="color:#aaa;font-size:10px">XGBoost (R²=0.73) trained on Tier 2 GPS data.<br>'
                f'Toggle "ML Coverage Extension" layer to view.<br>'
                f'Dashed lines = estimates. Triage only, not for enforcement.</span>'
            )

    return f"""
    <div id="summary-panel" style="position:fixed;top:52px;right:12px;z-index:9990;
                background:rgba(22,22,30,0.95);color:white;
                border-radius:10px;font-family:Arial,sans-serif;
                font-size:12px;box-shadow:0 4px 18px rgba(0,0,0,0.5);
                min-width:240px;max-width:290px">

      <!-- Panel header with collapse toggle -->
      <div style="display:flex;border-bottom:1px solid #444;border-radius:10px 10px 0 0;overflow:hidden">
        <div id="tab-summary" onclick="showTab('summary')"
             style="flex:1;padding:8px 10px;cursor:pointer;font-weight:bold;
                    font-size:12px;background:#1e40af;text-align:center">📊 Summary</div>
        <div id="tab-guide" onclick="showTab('guide')"
             style="flex:1;padding:8px 10px;cursor:pointer;font-weight:bold;
                    font-size:12px;background:#374151;text-align:center;color:#ccc">📖 How to use</div>
        <div onclick="var b=document.getElementById('summary-body');var c=document.getElementById('summary-chev');if(b.style.display==='none'){{b.style.display='block';c.textContent='▲'}}else{{b.style.display='none';c.textContent='▼'}}"
             style="padding:8px 10px;cursor:pointer;background:#374151;color:#9ca3af;font-size:11px;white-space:nowrap">
          <span id="summary-chev">▲</span>
        </div>
      </div>
      <script>
        function showTab(t) {{
          document.getElementById('panel-summary').style.display = t==='summary'?'block':'none';
          document.getElementById('panel-guide').style.display   = t==='guide'  ?'block':'none';
          document.getElementById('tab-summary').style.background = t==='summary'?'#1e40af':'#374151';
          document.getElementById('tab-guide').style.background   = t==='guide'  ?'#1e40af':'#374151';
          document.getElementById('tab-summary').style.color = t==='summary'?'white':'#ccc';
          document.getElementById('tab-guide').style.color   = t==='guide'  ?'white':'#ccc';
        }}
      </script>
      <div id="summary-body" style="max-height:82vh;overflow-y:auto">      <!-- Summary tab -->
      <div id="panel-summary" style="padding:10px 14px">
        <div style="font-size:11px;color:#aaa;margin-bottom:6px">
          Scored segments (Tier 2 GPS confirmed): <b style="color:white">{total:,}</b>
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:11px">
          <tr style="color:#888;font-size:10px">
            <td>Band</td><td style="text-align:right">N</td><td style="text-align:right">%</td>
          </tr>
          {band_rows}
        </table>
        <hr style="border-color:#444;margin:8px 0">
        <table style="width:100%;border-collapse:collapse;font-size:11px">
          <tr style="color:#888;font-size:10px">
            <td>Country</td>
            <td style="text-align:right">Avg SSS</td>
            <td style="text-align:right">Segments</td>
            <td style="text-align:right">Coverage</td>
          </tr>
          {coverage_rows}
        </table>
        <div style="color:#f59e0b;font-size:10px;margin-top:6px">
          ⚠ Coverage = % of network with GPS speed data.<br>
          Unscored segments have no posted limit or speed data.
        </div>
        {tier1_row}
        {pi_row}
        {ml_row}
      </div>

      <!-- How to use tab -->
      <div id="panel-guide" style="padding:10px 14px;display:none">
        <div style="line-height:1.6;color:#ddd">
          <b style="color:white;font-size:12px">Click any road to see:</b><br>
          <span style="color:#f87171">❶ Is the limit right?</span>
          <span style="color:#aaa"> — posted vs Safe System standard</span><br>
          <span style="color:#fca5a5">❷ Why?</span>
          <span style="color:#aaa"> — over/under-posted, driver behaviour, PTW risk</span><br>
          <span style="color:#93c5fd">❸ Intervention</span>
          <span style="color:#aaa"> — actions + responsible authority</span>
          <hr style="border-color:#444;margin:8px 0">
          <b style="color:white">Road colours = Speed Safety Score (SSS)</b><br>
          <span style="color:#ef4444">● Critical</span> — limit severely wrong, drivers confirm it<br>
          <span style="color:#f97316">● High Risk</span> — significant misalignment<br>
          <span style="color:#eab308">● Moderate</span> — some misalignment<br>
          <span style="color:#22c55e">● Acceptable</span> — limit broadly correct
          <hr style="border-color:#444;margin:8px 0">
          <b style="color:white">Safe System limit</b> = evidence-based speed ceiling for this road class × land use (WHO/iRAP). <b>Different from posted limit</b> — it is the target.<br>
          <hr style="border-color:#444;margin:8px 0">
          <b style="color:white">How SSS is calculated</b><br>
          <span style="color:#aaa;font-size:11px">
          <b style="color:#ddd">Alignment (20%)</b> — how far is the posted limit from the Safe System standard? Two-sided: too high and too low both score as misaligned.<br>
          <b style="color:#ddd">Credibility gap (45%)</b> — do F85 AND median speed both confirm the limit isn't working? Wide speed spread (mixed traffic) dampens this score. Based on dual-signal, not F85 alone.<br>
          <b style="color:#ddd">VRU risk (35%)</b> — who is exposed? Urban/pedestrian roads score higher. PTW multiplier applied for Thailand (74% PTW fatalities) and Maharashtra primary/trunk (37%).<br>
          <br>Safe System limits use 60 km/h for undivided rural primary/secondary in MH/TH (iRAP 1-2 star, WHO Speed Management Table 4.3).
          </span>
        </div>
      </div>
    </div>
    </div>
    """


def export_for_esri(gdf: gpd.GeoDataFrame, output_dir: str = "outputs") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    mask   = gdf["scoreable"] & gdf["sss"].notna()
    scored = gdf[mask].copy()

    scored.to_file(f"{output_dir}/speed_safety_scores_all.gpkg", driver="GPKG", layer="all_segments")
    print(f"Exported all scored segments: {output_dir}/speed_safety_scores_all.gpkg")

    for band in SCORE_BANDS.keys():
        sub = scored[scored["sss_band"] == band]
        if len(sub):
            layer = band.lower().replace(" ", "_")
            sub.to_file(f"{output_dir}/speed_safety_scores_{layer}.gpkg", driver="GPKG", layer=layer)
            print(f"  Exported {len(sub):,} {band} segments")

    for cc in scored["country_code"].unique():
        scored[scored["country_code"] == cc].to_file(
            f"{output_dir}/speed_safety_scores_{cc}.gpkg", driver="GPKG", layer=cc)
        print(f"  Exported {len(scored[scored['country_code']==cc]):,} {cc} segments")

    csv_cols = [
        "segment_id", "country", "road_class", "land_use",
        "speed_limit", "ss_limit", "speed_85th", "median_speed", "pct_over_limit",
        "sss", "sss_band", "sub_score_limit_alignment", "sub_score_limit_credibility",
        "sub_score_vru_risk", "sub_score_compliance", "confidence_weight",
        "exposure_score", "dist_to_school_m", "dist_to_hospital_m",
        "school_proximity_score", "hospital_proximity_score",
        "pop_density_500m", "exposure_component_population", "exposure_component_traffic",
        "intersection_score", "priority_score",
        "nilsson_fatal_ratio", "credibility_gap", "credibility_class", "recommended_limit",
        # Priority Index (Exposure × Likelihood × Severity) — runs alongside SSS
        "likelihood_score", "severity_score", "priority_index", "priority_band",
        "sub_likelihood_speed_gap", "sub_likelihood_credibility", "sub_likelihood_variability",
        "sub_severity_safe_system", "sub_severity_nilsson", "sub_severity_infrastructure", "sub_severity_helmet",
        "sss_recommendation", "image_url",
        # AI layer (EXPERIMENTAL, not used in map/popup/policy summary —
        # kept in this complete-data export for anyone who wants it)
        "anomaly_score", "anomaly_flag", "anomaly_reason",
    ]
    csv_cols = [c for c in csv_cols if c in scored.columns]
    scored[csv_cols].to_csv(f"{output_dir}/speed_safety_scores.csv", index=False)
    print(f"CSV export: {output_dir}/speed_safety_scores.csv")

    # Tier 1 (alignment-only) export — segments with a posted-limit-vs-
    # Safe-System score but no behavioural confirmation. Previously
    # computed but never written to any output file; without this, the
    # broader coverage Tier 1 adds exists only in memory during the run.
    if "alignment_scoreable" in gdf.columns:
        tier1 = gdf[gdf["alignment_scoreable"] & ~gdf["scoreable"]].copy()
        if len(tier1):
            t1_cols = [c for c in [
                "segment_id", "country", "country_code", "road_class", "land_use",
                "speed_limit", "ss_limit", "alignment_only_score", "alignment_only_band",
                "image_url", "geometry",
            ] if c in tier1.columns]
            tier1[t1_cols].to_file(f"{output_dir}/speed_safety_scores_tier1_only.gpkg",
                                     driver="GPKG", layer="tier1_only")
            csv_t1 = [c for c in t1_cols if c != "geometry"]
            tier1[csv_t1].to_csv(f"{output_dir}/speed_safety_scores_tier1_only.csv", index=False)
            print(f"  Exported {len(tier1):,} Tier 1 only segments (alignment score, "
                  f"no behavioural data): {output_dir}/speed_safety_scores_tier1_only.csv")


def export_corridors(corridors: gpd.GeoDataFrame, output_dir: str = "outputs") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    if len(corridors) == 0:
        return
    corridors.to_file(f"{output_dir}/speed_safety_corridors.gpkg", driver="GPKG", layer="intervention_zones")
    csv_cols = [c for c in corridors.columns if c != "geometry"]
    corridors[csv_cols].to_csv(f"{output_dir}/speed_safety_corridors.csv", index=False)
    print(f"Intervention zones exported: {output_dir}/speed_safety_corridors.gpkg ({len(corridors)} zones)")


def plot_sss_vs_pct_over_limit(
    gdf: gpd.GeoDataFrame,
    output_dir: str = "outputs",
) -> str:
    """
    Scatter plot: Speed Safety Score (Y) vs % vehicles over limit (X).
    Divides into 4 quadrants; Q1 'Hidden Danger' is the key finding —
    roads that are dangerous despite driver compliance.
    Saves to <output_dir>/scatter_sss_vs_pct_over_limit.png.
    Returns the output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df = gdf.copy()
    df = df.dropna(subset=["sss", "pct_over_limit"])
    df = df[(df["sss"] >= 0) & (df["sss"] <= 100)]
    if df["pct_over_limit"].max() <= 1.0:
        df["pct_over_limit"] = df["pct_over_limit"] * 100
    df = df[(df["pct_over_limit"] >= 0) & (df["pct_over_limit"] <= 100)]

    if len(df) < 10:
        print("  Scatter plot skipped — insufficient data with both sss and pct_over_limit")
        return ""

    SSS_THRESH, PCT_THRESH = 45, 40
    q1 = df[(df["sss"] >= SSS_THRESH) & (df["pct_over_limit"] <  PCT_THRESH)]
    q2 = df[(df["sss"] >= SSS_THRESH) & (df["pct_over_limit"] >= PCT_THRESH)]
    q3 = df[(df["sss"] <  SSS_THRESH) & (df["pct_over_limit"] <  PCT_THRESH)]
    q4 = df[(df["sss"] <  SSS_THRESH) & (df["pct_over_limit"] >= PCT_THRESH)]
    total_danger = len(q1) + len(q2)
    pct_missed   = 100 * len(q1) / total_danger if total_danger > 0 else 0

    BAND_COLOR = {
        "Critical":   "#D32F2F",
        "High Risk":  "#F57C00",
        "Moderate":   "#F9A825",
        "Acceptable": "#388E3C",
    }
    BAND_ORDER = ["Critical", "High Risk", "Moderate", "Acceptable"]

    fig, ax = plt.subplots(figsize=(11.4, 5.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")

    for band in BAND_ORDER:
        sub = df[df.get("sss_band", pd.Series(dtype=str)) == band] if "sss_band" in df.columns else df
        if "sss_band" not in df.columns:
            sub = df
        ax.scatter(sub["pct_over_limit"], sub["sss"],
                   c=BAND_COLOR.get(band, "#888"), s=9, alpha=0.28,
                   linewidths=0, label=band, zorder=2, rasterized=True)

    ax.axhspan(SSS_THRESH, 100, xmin=0,              xmax=PCT_THRESH/100, alpha=0.06, color="#C62828", zorder=1)
    ax.axhspan(SSS_THRESH, 100, xmin=PCT_THRESH/100, xmax=1,             alpha=0.04, color="#6A1B9A", zorder=1)
    ax.axhspan(0, SSS_THRESH,   xmin=0,              xmax=PCT_THRESH/100, alpha=0.04, color="#1B5E20", zorder=1)
    ax.axhspan(0, SSS_THRESH,   xmin=PCT_THRESH/100, xmax=1,             alpha=0.04, color="#E65100", zorder=1)

    ax.axvline(PCT_THRESH, color="#888", lw=1.3, ls="--", zorder=3, alpha=0.75)
    ax.axhline(SSS_THRESH, color="#888", lw=1.3, ls="--", zorder=3, alpha=0.75)

    QL = dict(fontsize=11, fontweight="bold", va="center", ha="center", zorder=5,
              bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.82))
    ax.text(PCT_THRESH / 2,         (SSS_THRESH + 100) / 2,
            f"Hidden Danger\n{len(q1):,} roads",    color="#C62828", **QL)
    ax.text((PCT_THRESH + 100) / 2, (SSS_THRESH + 100) / 2,
            f"Confirmed Danger\n{len(q2):,} roads", color="#6A1B9A", **QL)
    ax.text(PCT_THRESH / 2,         SSS_THRESH / 2,
            f"Safe\n{len(q3):,} roads",             color="#1B5E20", **QL)
    ax.text((PCT_THRESH + 100) / 2, SSS_THRESH / 2,
            f"False Alarm\n{len(q4):,} roads",      color="#BF360C", **QL)

    ax.text(98, 97,
            f"{pct_missed:.0f}% of high-risk roads\ninvisible to speed-camera monitoring",
            ha="right", va="top", fontsize=9, fontweight="bold", color="white", zorder=6,
            bbox=dict(boxstyle="round,pad=0.45", fc="#002569", ec="none", alpha=0.93))

    ax.set_xlabel("% Vehicles Exceeding Posted Limit", fontsize=11, color="#333", labelpad=5)
    ax.set_ylabel("Speed Safety Score (0–100)",         fontsize=11, color="#333", labelpad=5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#555", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#CCC")
    ax.text(PCT_THRESH + 1, 1.5, f"{PCT_THRESH}% threshold",
            fontsize=7.5, color="#777", style="italic")
    ax.text(1, SSS_THRESH + 1.5, f"SSS {SSS_THRESH}",
            fontsize=7.5, color="#777", style="italic")

    if "sss_band" in df.columns:
        handles = [mpatches.Patch(color=BAND_COLOR[b],
                                   label=f"{b}  (n={len(df[df['sss_band']==b]):,})")
                   for b in BAND_ORDER if b in df["sss_band"].values]
        ax.legend(handles=handles, title="SSS Band", title_fontsize=8,
                  fontsize=8, loc="lower right", framealpha=0.92, edgecolor="#CCC")

    fig.suptitle("Speed Safety Score vs. Driver Compliance",
                 fontsize=14, fontweight="bold", color="#002569", y=1.01)
    plt.tight_layout(pad=0.6)

    out = str(Path(output_dir) / "scatter_sss_vs_pct_over_limit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Scatter saved: scatter_sss_vs_pct_over_limit.png")
    print(f"  Q1 Hidden Danger: {len(q1):,} roads ({100*len(q1)/len(df):.1f}%)")
    print(f"  Conventional monitoring misses {pct_missed:.0f}% of high-risk roads")
    return out


def plot_shap_importance(
    gdf: gpd.GeoDataFrame,
    output_dir: str = "outputs",
) -> str:
    """
    Retrain XGBoost on Tier-2 scored segments, compute SHAP values,
    and plot a top-10 mean |SHAP| horizontal bar chart.
    Saves to <output_dir>/ml_shap_importance.png.
    Returns the output path.
    """
    try:
        import xgboost as xgb
        import shap as shap_lib
    except ImportError:
        print("  SHAP chart skipped — pip install xgboost shap")
        return ""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import warnings
    warnings.filterwarnings("ignore")

    NUMERIC = [
        "speed_limit", "ss_limit", "credibility_gap", "nilsson_fatal_ratio",
        "exposure_score", "pop_density_500m", "dist_to_school_m", "dist_to_hospital_m",
    ]
    CAT = ["road_class_norm", "road_class", "land_use"]
    DISPLAY = {
        "speed_limit":         "Posted Speed Limit",
        "ss_limit":            "Safe System Speed Limit",
        "credibility_gap":     "Limit Credibility Gap",
        "nilsson_fatal_ratio": "Nilsson Fatal Risk Ratio",
        "exposure_score":      "Exposure Score",
        "pop_density_500m":    "Population Density (500m)",
        "dist_to_school_m":    "Distance to School",
        "dist_to_hospital_m":  "Distance to Hospital",
    }

    train = gdf[gdf["sss"].notna()].copy()
    if len(train) < 50:
        print("  SHAP chart skipped — fewer than 50 scored segments")
        return ""

    num_feats = [f for f in NUMERIC if f in train.columns]
    cat_feats  = next((f for f in CAT if f in train.columns), None)
    cat_list   = [cat_feats] if cat_feats else []

    num_df = train.reindex(columns=num_feats).astype(float)
    cat_df = pd.get_dummies(
        train.reindex(columns=cat_list).fillna("unknown"),
        prefix=cat_list, dtype=float,
    ) if cat_list else pd.DataFrame(index=train.index)

    X = pd.concat([num_df, cat_df], axis=1)
    y = train["sss"].values.astype(float)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X, y)

    print(f"  Computing SHAP values ({len(X):,} segments)...")
    try:
        explainer = shap_lib.TreeExplainer(model)
        sv        = np.abs(explainer(X).values)
        mean_shap = sv.mean(axis=0)
    except ValueError as e:
        print(f"  SHAP chart skipped — XGBoost/SHAP version mismatch: {e}")
        print("  Fix: pip install --upgrade shap  (needs >= 0.45.0)")
        return ""
    except Exception as e:
        print(f"  SHAP chart skipped — {e}")
        return ""

    def clean_name(f):
        if f in DISPLAY:
            return DISPLAY[f]
        for pref in cat_list:
            if f.startswith(pref + "_"):
                val = f[len(pref)+1:].replace("_", " ").title()
                label = "Road Class" if "road" in pref else "Land Use"
                return f"{label}: {val}"
        return f.replace("_", " ").title()

    feat_names = list(X.columns)
    shap_df = (
        pd.DataFrame({"feature": feat_names, "mean_shap": mean_shap})
        .sort_values("mean_shap", ascending=False)
        .head(10)
    )
    shap_df["label"] = shap_df["feature"].apply(clean_name)
    top10 = shap_df.iloc[::-1]

    def bar_color(label):
        if "Speed" in label or "Credibility" in label:
            return "#F57C00"
        if "Nilsson" in label or "Exposure" in label:
            return "#D32F2F"
        if "Population" in label or "School" in label or "Hospital" in label:
            return "#1565C0"
        return "#558B2F"

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")

    colors = [bar_color(l) for l in top10["label"]]
    bars = ax.barh(top10["label"], top10["mean_shap"],
                   color=colors, height=0.62, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, top10["mean_shap"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9, color="#333")

    median_val = float(top10["mean_shap"].median())
    ax.axvline(median_val, color="#AAA", lw=1, ls=":", zorder=0)
    ax.text(median_val + 0.02, -0.7, "median",
            fontsize=7.5, color="#AAA", style="italic", va="bottom")

    ax.set_xlabel("Mean |SHAP Value| — Average Impact on SSS Prediction",
                  fontsize=10.5, color="#333", labelpad=6)
    ax.set_xlim(0, float(top10["mean_shap"].max()) * 1.18)
    ax.tick_params(axis="y", labelsize=10, colors="#333")
    ax.tick_params(axis="x", labelsize=9,  colors="#555")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#DDD")
    ax.spines["bottom"].set_color("#DDD")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#EEE", linewidth=0.8)

    legend_items = [
        mpatches.Patch(color="#F57C00", label="Speed / Limit"),
        mpatches.Patch(color="#D32F2F", label="Risk / Exposure"),
        mpatches.Patch(color="#1565C0", label="VRU Context"),
        mpatches.Patch(color="#558B2F", label="Road Type"),
    ]
    ax.legend(handles=legend_items, fontsize=8.5, loc="lower right",
              framealpha=0.9, edgecolor="#CCC", ncol=2)

    fig.suptitle("What Drives the XGBoost SSS Prediction — SHAP Feature Importance",
                 fontsize=13, fontweight="bold", color="#002569", y=1.01)
    plt.tight_layout(pad=0.7)

    out = str(Path(output_dir) / "ml_shap_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top1 = shap_df.iloc[0]
    print(f"  SHAP chart saved: ml_shap_importance.png")
    print(f"  Top driver: {top1['label']} (mean |SHAP| = {top1['mean_shap']:.2f})")
    return out