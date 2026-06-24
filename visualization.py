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
    Four-layer popup per ADB Challenge brief:
      Layer 1 (Executive):    Band + SSS score + speed comparison grid + confidence bar
      Layer 2 (Explanation):  Plain-language reasons with evidence citations
      Layer 3 (Intervention): Specific actions + impact + authority
      Layer 4 (Technical):    Sub-scores, Nilsson range, data sources — collapsed
    Design principles:
      - Policymaker readable in <10 seconds (top card only)
      - Every AI output shows what drove it (Layer 2 evidence bullets)
      - No black-box: data sources shown in Layer 4
      - Progressive disclosure: jargon hidden until requested
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

    BAND_BG = {
        "Critical":    "#b91c1c", "High Risk": "#c2410c",
        "Moderate":    "#a16207", "Acceptable": "#15803d",
    }
    BAND_LIGHT = {
        "Critical":    "#fef2f2", "High Risk": "#fff7ed",
        "Moderate":    "#fefce8", "Acceptable": "#f0fdf4",
    }
    BAND_BADGE = {
        "Critical": "#ef4444", "High Risk": "#f97316",
        "Moderate": "#eab308", "Acceptable": "#22c55e",
    }

    band_bg    = BAND_BG.get(band, "#374151")
    band_badge = BAND_BADGE.get(band, "#9ca3af")

    def fmt(v, unit="", dec=1):
        return f"{v:.{dec}f}{unit}" if pd.notna(v) else "—"

    # ── LAYER 1: Executive card ───────────────────────────────────────────────
    sss_pct = min(100, max(0, sss)) if pd.notna(sss) else 0

    # Limit verdict
    if pd.notna(sl) and pd.notna(ss):
        gap = sl - ss
        if sl > ss + 5:
            verdict       = "Too high"
            verdict_color = "#b91c1c"
            limit_detail  = f"Posted {sl:.0f} km/h — Safe System standard is {ss:.0f} km/h (+{gap:.0f} km/h over)"
        elif sl < ss * 0.80:
            verdict       = "Too low — likely outdated"
            verdict_color = "#c2410c"
            limit_detail  = f"Posted {sl:.0f} km/h — Safe System standard is {ss:.0f} km/h (limit {ss-sl:.0f} km/h below design speed)"
        elif sl < ss - 5:
            verdict       = "Slightly low"
            verdict_color = "#a16207"
            limit_detail  = f"Posted {sl:.0f} km/h — slightly below Safe System standard ({ss:.0f} km/h)"
        else:
            verdict       = "Broadly appropriate"
            verdict_color = "#15803d"
            limit_detail  = f"Posted {sl:.0f} km/h aligns with Safe System standard ({ss:.0f} km/h)"
    else:
        verdict       = "No data"
        verdict_color = "#6b7280"
        limit_detail  = "No posted speed limit recorded for this segment"

    # Speed comparison grid
    speed_grid = ""
    if pd.notna(sl) or pd.notna(ss) or pd.notna(med):
        def _cell(label, val, highlight=False, warn=False):
            bg    = "#eff6ff" if highlight else ("#fef2f2" if warn else "#f9fafb")
            border= "#bfdbfe" if highlight else ("#fecaca" if warn else "#e5e7eb")
            color = "#1d4ed8" if highlight else ("#b91c1c" if warn else "#111827")
            v     = fmt(val, " km/h", 0) if pd.notna(val) else "—"
            return (f'<td style="padding:0 4px;text-align:center">'
                    f'<div style="background:{bg};border:1px solid {border};border-radius:6px;padding:5px 8px">'
                    f'<div style="font-size:10px;color:#6b7280;margin-bottom:2px">{label}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:{color}">{v}</div>'
                    f'</div></td>')

        speed_grid = (
            '<table style="width:100%;border-collapse:collapse;margin-top:8px">'
            '<tr>'
            + _cell("Posted limit", sl, highlight=True)
            + _cell("Safe System", ss)
            + _cell("Median (GPS)", med, warn=(pd.notna(med) and pd.notna(sl) and med > sl))
            + _cell("F85 (GPS)", f85, warn=(pd.notna(f85) and pd.notna(sl) and f85 > sl + 10))
            + '</tr></table>'
        )
        if pd.notna(spread):
            spread_warn = ' style="color:#c2410c"' if spread > 20 else ""
            speed_grid += (f'<div style="font-size:10px;color:#6b7280;margin-top:5px">'
                           f'Speed spread (F85−median): <b{spread_warn}>{spread:.0f} km/h</b>'
                           + (" — wide spread suggests mixed traffic; F85 may not represent typical driver" if spread > 20 else "")
                           + '</div>')

    # Confidence / SSS bar
    confidence_bar = (
        f'<div style="margin-top:8px">'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:rgba(255,255,255,0.7);margin-bottom:3px">'
        f'<span>Risk Score</span><span style="font-weight:700">{sss_pct:.1f} / 100</span></div>'
        f'<div style="height:5px;background:rgba(255,255,255,0.2);border-radius:3px;overflow:hidden">'
        f'<div style="height:100%;width:{sss_pct}%;background:rgba(255,255,255,0.85);border-radius:3px"></div>'
        f'</div></div>'
    )

    # ── LAYER 2: Explanation bullets ─────────────────────────────────────────
    reasons = []

    def _reason(icon, text):
        return (f'<div style="display:flex;align-items:flex-start;gap:8px;'
                f'background:#fff;border:1px solid #e5e7eb;'
                f'border-left:3px solid #ef4444;border-radius:0 6px 6px 0;'
                f'padding:7px 10px;margin-bottom:5px">'
                f'<span style="color:#ef4444;font-size:13px;flex-shrink:0">{icon}</span>'
                f'<span style="font-size:12px;color:#374151;line-height:1.5">{text}</span>'
                f'</div>')

    if pd.notna(sl) and pd.notna(ss) and sl > ss + 5:
        reasons.append(_reason("↑", f"Limit {sl:.0f} km/h exceeds {ss:.0f} km/h Safe System ceiling for {lu} {rc}"))
    if pd.notna(sl) and pd.notna(ss) and sl < ss * 0.80:
        reasons.append(_reason("⟳", f"Limit {sl:.0f} km/h is {ss-sl:.0f} km/h below road design speed — likely outdated or never revised after road upgrade"))
    if pd.notna(f85) and pd.notna(sl) and pd.notna(med) and med > sl and f85 > sl + 15:
        reasons.append(_reason("🚗", f"Typical driver (median {med:.0f} km/h) AND fast tail (F85 {f85:.0f} km/h) both exceed posted limit — systemic non-compliance, not an outlier"))
    elif pd.notna(f85) and pd.notna(sl) and f85 > sl + 15:
        reasons.append(_reason("🚗", f"F85 {f85:.0f} km/h significantly exceeds posted limit by {f85-sl:.0f} km/h"))
    if credibility == "Non-Credible":
        reasons.append(_reason("⚠", "Limit is non-credible — F85 >20 km/h above posted means drivers have effectively stopped following signage"))
    if credibility == "Infrastructure-Forced":
        reasons.append(_reason("🔍", "Low speeds likely caused by speed bumps or tables — verify infrastructure before recommending limit change"))
    if pd.notna(spread) and spread > 20:
        reasons.append(_reason("⇔", f"Wide speed distribution ({spread:.0f} km/h spread) — mixed traffic (trucks, PTW, cars); GPS probe data is car-biased"))
    if pd.notna(nilsson) and nilsson > 4:
        nl_str = f" (Asian range: {fmt(nilsson_low,'×',1)}–{fmt(nilsson_high,'×',1)})" if pd.notna(nilsson_low) else ""
        reasons.append(_reason("❤", f"Fatal crash risk {nilsson:.1f}× Safe System baseline (Nilsson Power Model){nl_str}"))
    if pd.notna(sinuosity) and sinuosity >= 1.50:
        reasons.append(_reason("↩", f"Sharply curved alignment (sinuosity {sinuosity:.2f}) — sight distance limits safe speed"))
    elif pd.notna(sinuosity) and sinuosity >= 1.20:
        reasons.append(_reason("↩", f"Curved alignment (sinuosity {sinuosity:.2f}) — design speed adjustment applied"))
    if cc == "TH" and rc in ("secondary", "tertiary", "primary"):
        reasons.append(_reason("🏍", "PTW high-risk corridor — Thailand PTW riders account for 74% of road fatalities (WHO 2023)"))
    if cc == "MH" and rc in ("primary", "trunk"):
        reasons.append(_reason("🏍", "PTW–truck conflict zone — Maharashtra PTW 37% of fatalities; undivided carriageway (iRAP 1–2 star)"))
    if not reasons:
        reasons.append(_reason("✓", "Limit broadly appropriate; no major credibility or alignment issues detected"))

    reasons_html = "".join(reasons)

    # ── LAYER 3: Intervention actions ─────────────────────────────────────────
    actions = []

    def _action(icon, label, detail, impact, priority="Recommended"):
        p_color = "#b91c1c" if priority == "Priority" else "#c2410c" if priority == "Urgent" else "#374151"
        return (f'<div style="background:#eff6ff;border:1px solid #bfdbfe;'
                f'border-left:3px solid #3b82f6;border-radius:0 6px 6px 0;'
                f'padding:8px 12px;margin-bottom:6px">'
                f'<div style="display:flex;align-items:flex-start;gap:8px">'
                f'<span style="font-size:14px;color:#3b82f6;flex-shrink:0">{icon}</span>'
                f'<div style="flex:1">'
                f'<div style="font-size:12px;font-weight:700;color:#1e3a5f;margin-bottom:2px">{label}</div>'
                f'<div style="font-size:11px;color:#4b5563;margin-bottom:3px">{detail}</div>'
                f'<div style="font-size:10px;color:#059669">↑ {impact}</div>'
                f'</div>'
                f'<span style="font-size:10px;font-weight:700;color:{p_color};'
                f'background:{p_color}12;border-radius:4px;padding:2px 6px;white-space:nowrap">{priority}</span>'
                f'</div></div>')

    if pd.notna(sl) and pd.notna(ss) and sl > ss + 5:
        rl = rec_limit if pd.notna(rec_limit) else ss
        reduction = sl - rl
        effort = row.get("change_effort", "")
        actions.append(_action("🚦", f"Reduce speed limit to {rl:.0f} km/h (−{reduction:.0f} km/h)",
            f"Recommended limit based on Safe System standard for {lu} {rc} ({effort})",
            "20–40% reduction in fatal crash probability (Nilsson Power Model)", "Priority"))
    if pd.notna(sl) and pd.notna(ss) and sl < ss * 0.80:
        actions.append(_action("📋", f"Commission road audit before raising limit",
            f"Posted {sl:.0f} km/h appears outdated vs Safe System {ss:.0f} km/h. Audit confirms design speed.",
            "Road audit prevents inappropriate limit increases on substandard infrastructure", "Priority"))
    if credibility == "Non-Credible":
        actions.append(_action("📷", "Install physical calming or speed enforcement cameras",
            "Signage alone is not working — F85 >20 km/h above posted limit",
            "Speed cameras reduce F85 by 5–15 km/h on treated corridors (WHO evidence)", "Urgent"))
    if credibility == "Infrastructure-Forced":
        actions.append(_action("🔍", "Verify speed bump / table presence on-site",
            "Low F85 + low median + tight spread = probable infrastructure forcing",
            "Prevents incorrect 'raise the limit' recommendation on bump-controlled roads", "Recommended"))
    if pd.notna(nilsson) and nilsson > 4 and rc in ("trunk", "primary"):
        actions.append(_action("🛡", "Install median barrier or physical separation",
            f"Fatal crash risk {nilsson:.1f}× baseline on undivided carriageway",
            "Median barriers reduce head-on fatalities by ~50% (iRAP evidence base)", "Priority"))
    if pd.notna(sinuosity) and sinuosity >= 1.50:
        actions.append(_action("⚠", "Install curve warning chevrons and advance warning signs",
            f"Sinuosity {sinuosity:.2f} — sight distance limits safe speed on this alignment",
            "Curve warning signs reduce curve crashes 15–25% (PIARC evidence)", "Recommended"))
    elif pd.notna(sinuosity) and sinuosity >= 1.20:
        actions.append(_action("⚠", "Install curve advisory speed signs",
            f"Sinuosity {sinuosity:.2f} — moderate curvature detected",
            "Advisory signs alert drivers to design speed limitation", "Recommended"))
    if cc == "TH" and rc in ("secondary", "tertiary", "primary"):
        actions.append(_action("🏍", "Deploy PTW-targeted enforcement and awareness campaign",
            "Thailand PTW 74% of fatalities — this corridor type is documented high-risk",
            "PTW-specific enforcement reduces PTW fatalities 20–35% (SWOV evidence)", "Recommended"))
    if cc == "MH" and rc in ("primary", "trunk"):
        actions.append(_action("🏍", "Install rumble strips and hard shoulder demarcation",
            "Edge-line rumble strips alert PTW riders drifting toward opposing lane; "
            "shoulder demarcation clarifies carriageway boundary on roads with no kerb",
            "PTW lane departure interventions reduce PTW fatalities 20–35% (SWOV)", "Recommended"))
    if not actions:
        actions.append(_action("📊", "Monitor — schedule next audit in 12 months",
            "No high-priority issues detected on this segment",
            "Routine monitoring maintains data currency", "Routine"))

    authority_map_mh = {
        "motorway": "NHAI", "trunk": "NHAI / MSRDC",
        "primary": "Maharashtra PWD", "secondary": "Maharashtra PWD / District",
        "tertiary": "District / Municipal Authority",
    }
    authority_map_th = {
        "motorway": "DOH — Dept. of Highways", "trunk": "DOH — Dept. of Highways",
        "primary": "DOH — Dept. of Highways", "secondary": "DRR — Dept. of Rural Roads",
        "tertiary": "DRR / Local Administration",
    }
    authority = (authority_map_mh if cc == "MH" or "Maharashtra" in country else authority_map_th).get(rc, "Road Authority")
    actions_html = "".join(actions)

    tech_id  = f"tech_{str(seg_id).replace('/','_').replace(' ','_')}"
    tab1_id  = f"t1_{tech_id}"
    tab2_id  = f"t2_{tech_id}"
    tab1b_id = f"tb1_{tech_id}"
    tab2b_id = f"tb2_{tech_id}"

    # Priority Index badge for metrics tab
    pi_html = ""
    if pd.notna(priority_index):
        pi_colors = {
            "Critical":   "#fee2e2:#991b1b",
            "High Risk":  "#ffedd5:#9a3412",
            "Moderate":   "#fef9c3:#713f12",
            "Acceptable": "#dcfce7:#14532d",
        }
        pi_c = pi_colors.get(priority_band, "#f3f4f6:#374151").split(":")
        pi_html = (f'<div style="margin-bottom:8px;padding:6px 10px;background:{pi_c[0]};'
                   f'border-radius:5px;font-size:11px;font-weight:600;color:{pi_c[1]}">'
                   f'Priority Index (Exposure × Likelihood × Severity): '
                   f'{priority_index:.1f} — {priority_band}</div>')

    nilsson_range = ""
    if pd.notna(nilsson_low) and pd.notna(nilsson_high):
        nilsson_range = f"<span style='color:#9ca3af'> [{nilsson_low:.1f}–{nilsson_high:.1f}×]</span>"

    data_sources = (
        "GPS mobility probe (ADB WeightedSample) · "
        "Safe System: WHO Speed Management Manual 2nd ed. + iRAP 1-2 star · "
        "Nilsson (2004) exponent 4.0 (Elvik 2009; Imprialou &amp; Quddus 2019) · "
        "PTW weights: WHO Global Status Report 2023"
    )

    img_html = ""
    if img_url and isinstance(img_url, str) and img_url.startswith("http"):
        img_html = (f'<div style="padding-top:8px;border-top:1px solid #f3f4f6">'
                    f'<a href="{img_url}" target="_blank" '
                    f'style="font-size:11px;color:#3b82f6;text-decoration:none">'
                    f'📷 View street imagery — verify road conditions</a></div>')

    # Score bar helper for metrics tab
    def _score_bar(score, weight_pct, color="#3b82f6"):
        w = min(100, max(0, score)) if pd.notna(score) else 0
        return (f'<div style="margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:11px;color:#6b7280;margin-bottom:3px">'
                f'<span>{weight_pct}</span>'
                f'<span style="font-weight:700;color:#111827">{fmt(score)}/100</span></div>'
                f'<div style="height:7px;background:#e5e7eb;border-radius:4px;overflow:hidden">'
                f'<div style="height:100%;width:{w}%;background:{color};border-radius:4px"></div>'
                f'</div></div>')

    pct_over = row.get("pct_over_limit", np.nan)
    change_effort = row.get("change_effort", "")
    weighted_sample = row.get("weighted_sample", row.get("sample_size", np.nan))
    est_lives = row.get("est_lives_saved", np.nan)
    nilsson_high_val = row.get("nilsson_fatal_ratio_high", np.nan)
    nilsson_low_val  = row.get("nilsson_fatal_ratio_low", np.nan)

    # Tab switching JS (unique per popup via IDs)
    tab_js = (
        f"function swTab_{tech_id}(t){{"
        f"var a=document.getElementById('{tab1_id}');"
        f"var b=document.getElementById('{tab2_id}');"
        f"var ta=document.getElementById('{tab1b_id}');"
        f"var tb=document.getElementById('{tab2b_id}');"
        f"a.style.display=t==1?'block':'none';"
        f"b.style.display=t==2?'block':'none';"
        f"ta.style.background=t==1?'#1d4ed8':'transparent';"
        f"ta.style.color=t==1?'#fff':'#6b7280';"
        f"tb.style.background=t==2?'#1d4ed8':'transparent';"
        f"tb.style.color=t==2?'#fff':'#6b7280';"
        f"}}"
    )

    return f"""
    <div style="font-family:system-ui,sans-serif;width:430px;background:#fff;
                border:1px solid #e5e7eb;border-radius:10px;overflow:hidden">
    <script>{tab_js}</script>

      <!-- HEADER: always visible -->
      <div style="background:{band_bg};color:#fff;padding:10px 16px">
        <div style="font-size:11px;opacity:0.85;margin-bottom:2px">
          {country} · {lu} {rc} · {seg_id}
        </div>
        <div style="display:flex;align-items:center;gap:10px">
          <div style="font-size:16px;font-weight:700">{band} — SSS {fmt(sss)}/100</div>
          <div style="margin-left:auto;font-size:11px;font-weight:700;
                      background:rgba(255,255,255,0.18);border-radius:20px;padding:3px 11px">
            {verdict}
          </div>
        </div>
        {confidence_bar}
      </div>

      <!-- Speed grid: always visible -->
      <div style="padding:10px 14px 0;border-bottom:1px solid #e5e7eb">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:4px;letter-spacing:0.04em;text-transform:uppercase">Speed at a glance</div>
        {speed_grid}
        <div style="margin-top:6px;font-size:11px;color:#374151;
                    background:#f9fafb;border-radius:5px;padding:5px 8px;margin-bottom:10px">
          {limit_detail}
        </div>
      </div>

      <!-- TAB BAR -->
      <div style="display:flex;border-bottom:1px solid #e5e7eb;background:#f9fafb">
        <button id="{tab1b_id}" onclick="swTab_{tech_id}(1)"
                style="flex:1;padding:8px 12px;border:none;cursor:pointer;font-size:12px;
                       font-weight:600;background:#1d4ed8;color:#fff;border-radius:0">
          ❶ Why &amp; What to do
        </button>
        <button id="{tab2b_id}" onclick="swTab_{tech_id}(2)"
                style="flex:1;padding:8px 12px;border:none;cursor:pointer;font-size:12px;
                       font-weight:600;background:transparent;color:#6b7280;border-radius:0">
          📊 Score metrics
        </button>
      </div>

      <!-- TAB 1: WHY + INTERVENTION -->
      <div id="{tab1_id}" style="display:block;padding:12px 14px">

        <div style="font-size:10px;font-weight:700;color:#6b7280;letter-spacing:0.06em;
                    text-transform:uppercase;margin-bottom:6px">Why this road is flagged</div>
        {reasons_html}

        <div style="display:flex;align-items:center;gap:8px;margin:10px 0 6px">
          <div style="font-size:10px;font-weight:700;color:#6b7280;letter-spacing:0.06em;text-transform:uppercase">
            Recommended interventions
          </div>
          <div style="margin-left:auto;font-size:10px;color:#6b7280;
                      background:#f3f4f6;border-radius:4px;padding:2px 8px;white-space:nowrap">
            {authority}
          </div>
        </div>
        {actions_html}
        {img_html}
      </div>

      <!-- TAB 2: SCORE METRICS -->
      <div id="{tab2_id}" style="display:none;padding:12px 14px">

        {pi_html}

        <!-- SSS Sub-scores with bars -->
        <div style="font-size:10px;font-weight:700;color:#6b7280;letter-spacing:0.06em;
                    text-transform:uppercase;margin-bottom:8px">SSS component scores</div>

        <div style="margin-bottom:4px;font-size:11px;color:#374151;font-weight:600">
          Alignment <span style="font-size:10px;font-weight:400;color:#9ca3af">(posted limit vs Safe System standard)</span>
        </div>
        {_score_bar(align_score, "Weight: 20%", "#3b82f6")}

        <div style="margin-bottom:4px;font-size:11px;color:#374151;font-weight:600">
          Credibility gap <span style="font-size:10px;font-weight:400;color:#9ca3af">(dual-signal F85 + median)</span>
        </div>
        {_score_bar(cred_score, "Weight: 45%", "#ef4444")}

        <div style="margin-bottom:4px;font-size:11px;color:#374151;font-weight:600">
          VRU context risk <span style="font-size:10px;font-weight:400;color:#9ca3af">(with PTW weighting)</span>
        </div>
        {_score_bar(vru_score, "Weight: 35%", "#f97316")}

        <!-- SSS formula result -->
        <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:6px;
                    padding:7px 10px;margin:8px 0;font-size:11px">
          <b style="color:#15803d">SSS = 0.20×{fmt(align_score)} + 0.45×{fmt(cred_score)} + 0.35×{fmt(vru_score)} = <span style="font-size:13px">{fmt(sss)}</span></b>
          <span style="color:#6b7280;margin-left:4px">→ {band}</span>
        </div>

        <hr style="border:none;border-top:1px solid #f3f4f6;margin:10px 0">

        <!-- Behaviour evidence -->
        <div style="font-size:10px;font-weight:700;color:#6b7280;letter-spacing:0.06em;
                    text-transform:uppercase;margin-bottom:8px">GPS behaviour evidence</div>

        <table style="width:100%;font-size:11px;border-collapse:collapse">
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">Posted limit</td>
            <td style="padding:4px 6px;font-weight:700;color:#1d4ed8;text-align:right">{fmt(sl,' km/h',0)}</td>
          </tr>
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">Safe System standard</td>
            <td style="padding:4px 6px;font-weight:700;color:#374151;text-align:right">{fmt(ss,' km/h',0)}</td>
          </tr>
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">Median speed (GPS probe)</td>
            <td style="padding:4px 6px;font-weight:700;color:{"#b91c1c" if pd.notna(med) and pd.notna(sl) and med > sl else "#374151"};text-align:right">{fmt(med,' km/h',0)}</td>
          </tr>
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">85th pct speed (F85)</td>
            <td style="padding:4px 6px;font-weight:700;color:{"#b91c1c" if pd.notna(f85) and pd.notna(sl) and f85 > sl+10 else "#374151"};text-align:right">{fmt(f85,' km/h',0)}</td>
          </tr>
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">Speed spread (F85−median)</td>
            <td style="padding:4px 6px;font-weight:700;color:{"#c2410c" if pd.notna(spread) and spread>20 else "#374151"};text-align:right">
              {fmt(spread,' km/h',0)}{"  ⚠ mixed traffic" if pd.notna(spread) and spread > 20 else ""}</td>
          </tr>
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">% vehicles over limit</td>
            <td style="padding:4px 6px;font-weight:700;color:#374151;text-align:right">{fmt(pct_over,'%',1)}</td>
          </tr>
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">Credibility class</td>
            <td style="padding:4px 6px;font-weight:700;color:#374151;text-align:right">{credibility if credibility else "—"}</td>
          </tr>
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">Sinuosity index</td>
            <td style="padding:4px 6px;font-weight:700;color:#374151;text-align:right">{fmt(sinuosity)} <span style="color:#9ca3af;font-weight:400">(1.0=straight)</span></td>
          </tr>
          <tr style="border-bottom:1px solid #f3f4f6">
            <td style="padding:4px 6px;color:#6b7280">Nilsson fatal risk ratio</td>
            <td style="padding:4px 6px;font-weight:700;color:{"#b91c1c" if pd.notna(nilsson) and nilsson>4 else "#374151"};text-align:right">
              {fmt(nilsson,'×')}{nilsson_range}</td>
          </tr>
          {"<tr style='border-bottom:1px solid #f3f4f6'><td style='padding:4px 6px;color:#6b7280'>Est. lives saved/yr (illustrative)</td><td style='padding:4px 6px;font-weight:700;color:#059669;text-align:right'>" + fmt(est_lives) + "</td></tr>" if pd.notna(est_lives) else ""}
          {"<tr><td style='padding:4px 6px;color:#6b7280'>Limit change effort</td><td style='padding:4px 6px;font-weight:700;color:#374151;text-align:right'>" + change_effort + "</td></tr>" if change_effort else ""}
        </table>

        <div style="margin-top:8px;padding:6px 8px;background:#f9fafb;
                    border-radius:5px;font-size:9px;color:#9ca3af;line-height:1.6">
          <b style="color:#6b7280">Sources:</b> {data_sources}
        </div>
      </div>

    </div>
    """

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
    max_segments: int = 2000,
    data_dir: str = "enrichment_data",
    max_amenity_markers: int = 6000,
) -> folium.Map:
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

    # Segment layers per country
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

            popup_html  = _build_popup_html(row)
            tooltip_txt = f"{band} | SSS: {sss:.1f}" if pd.notna(sss) else "No data"

            try:
                if geom.geom_type in ("LineString", "MultiLineString"):
                    for seg_coords in _geom_to_latlon_list(geom):
                        folium.PolyLine(
                            locations=seg_coords, color=color, weight=weight,
                            opacity=0.85,
                            popup=folium.Popup(popup_html, max_width=360),
                            tooltip=tooltip_txt,
                        ).add_to(fg)
                else:
                    c = geom.centroid
                    folium.CircleMarker(
                        location=[c.y, c.x], radius=5, color=color,
                        fill=True, fill_color=color, fill_opacity=0.8,
                        popup=folium.Popup(popup_html, max_width=360),
                        tooltip=tooltip_txt,
                    ).add_to(fg)
            except Exception:
                pass
        fg.add_to(m)

    # Tier 1 only — segments with a posted-limit-vs-Safe-System score but
    # no behavioural confirmation (no F85/median). Shown dashed/thin and
    # off by default so the broader (if lower-confidence) coverage this
    # tier adds is visible without competing visually with the main view.
    if len(tier1_only):
        fg_t1 = folium.FeatureGroup(
            name=f"📏 Tier 1 Only — Limit vs Standard ({len(tier1_only):,} segments)",
            show=False,
        )
        plot_t1 = tier1_only
        if len(tier1_only) > max_segments:
            plot_t1 = tier1_only.sample(max_segments, random_state=42)
        for _, row in plot_t1.iterrows():
            try:
                geom  = row.geometry
                score = row.get("alignment_only_score", np.nan)
                band  = row.get("alignment_only_band", "Acceptable")
                color = score_to_color(score)
                popup_html  = _build_popup_html(row)
                tooltip_txt = f"Tier 1 only | {band}: {score:.1f}" if pd.notna(score) else "No data"
                if geom.geom_type in ("LineString", "MultiLineString"):
                    for seg_coords in _geom_to_latlon_list(geom):
                        folium.PolyLine(
                            locations=seg_coords, color=color, weight=1.5,
                            opacity=0.6, dash_array="4,4",
                            popup=folium.Popup(popup_html, max_width=360),
                            tooltip=tooltip_txt,
                        ).add_to(fg_t1)
            except Exception:
                pass
        fg_t1.add_to(m)

    # Critical segments layer
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

    # Priority Index View — same segments, recolored by priority_index/
    # priority_band instead of sss/sss_band. Off by default (consistent with
    # the other secondary layers above); toggle it on via the layer control
    # to visually compare against the default SSS-colored view. This is the
    # main tool for "decide after seeing it" on the map itself.
    if "priority_index" in scored.columns:
        pi_scored = scored[scored["priority_index"].notna()]
        if len(pi_scored):
            fg_pi = folium.FeatureGroup(name="🎯 Priority Index View", show=False)
            plot_pi = pi_scored
            if len(pi_scored) > max_segments:
                plot_pi = pi_scored.sample(max_segments, random_state=42)

            for _, row in plot_pi.iterrows():
                try:
                    geom  = row.geometry
                    pi    = row.get("priority_index", np.nan)
                    band  = row.get("priority_band", "Acceptable")
                    color = score_to_color(pi)
                    weight = 3 if band in ("Critical", "High Risk") else 2

                    popup_html  = _build_popup_html(row)
                    tooltip_txt = f"{band} | Priority Index: {pi:.1f}" if pd.notna(pi) else "No data"

                    if geom.geom_type in ("LineString", "MultiLineString"):
                        for seg_coords in _geom_to_latlon_list(geom):
                            folium.PolyLine(
                                locations=seg_coords, color=color, weight=weight,
                                opacity=0.85,
                                popup=folium.Popup(popup_html, max_width=360),
                                tooltip=tooltip_txt,
                            ).add_to(fg_pi)
                    else:
                        c = geom.centroid
                        folium.CircleMarker(
                            location=[c.y, c.x], radius=5, color=color,
                            fill=True, fill_color=color, fill_opacity=0.8,
                            popup=folium.Popup(popup_html, max_width=360),
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
                min-width:240px;max-width:290px;max-height:86vh;overflow-y:auto">

      <!-- Tab header -->
      <div style="display:flex;border-bottom:1px solid #444">
        <div id="tab-summary" onclick="showTab('summary')"
             style="flex:1;padding:8px 10px;cursor:pointer;font-weight:bold;
                    font-size:12px;background:#1e40af;text-align:center;
                    border-radius:10px 0 0 0">📊 Summary</div>
        <div id="tab-guide" onclick="showTab('guide')"
             style="flex:1;padding:8px 10px;cursor:pointer;font-weight:bold;
                    font-size:12px;background:#374151;text-align:center;
                    border-radius:0 10px 0 0;color:#ccc">📖 How to use</div>
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

      <!-- Summary tab -->
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
