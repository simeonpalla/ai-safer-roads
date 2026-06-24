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


def _stag(row: pd.Series, col: str) -> str:
    """Safely extract a string tag column — handles pd.NA, None, and 'nan'/'None' strings."""
    v = row.get(col)
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return "" if s in ("nan", "None", "<NA>", "NaT", "none") else s


def _build_reason_html(row: pd.Series) -> str:
    """
    Plain-language reason block driven by F85/median/OSM/VIIRS signals.
    No threshold labels in the text — every sentence traces to an observed data point.
    """
    posted = row.get("speed_limit", np.nan)
    f85    = row.get("speed_85th",  np.nan)
    med    = row.get("median_speed", np.nan)

    if pd.isna(f85) or pd.isna(posted):
        return ""

    osm_lit     = _stag(row, "osm_lit").lower()
    osm_surface = _stag(row, "osm_surface").lower()
    osm_oneway  = _stag(row, "osm_oneway").lower()
    try:
        osm_lanes = float(row.get("osm_lanes") or 0)
    except (ValueError, TypeError):
        osm_lanes = 0
    ntl = row.get("ntl_exposure_score", np.nan)

    _UNPAVED = {"unpaved","gravel","dirt","ground","sand","earth",
                "laterite","compacted","fine_gravel"}

    gap = posted - f85  # positive = limit above F85 (overposted)

    if gap > 15:
        med_txt = f", median <b>{med:.0f} km/h</b>" if pd.notna(med) else ""
        primary = (
            f"Drivers travel at <b>{f85:.0f} km/h</b> (F85{med_txt}) — "
            f"<b>{gap:.0f} km/h</b> below the posted limit of <b>{posted:.0f} km/h</b>. "
            f"The posted limit is not calibrated to how this road is actually used."
        )
        icon, color = "⚠", "#c0392b"

    elif f85 - posted > 15:
        if pd.notna(med) and med > posted:
            primary = (
                f"Both F85 (<b>{f85:.0f} km/h</b>) and median (<b>{med:.0f} km/h</b>) "
                f"exceed the posted <b>{posted:.0f} km/h</b> limit — "
                f"not just outliers, the majority of drivers are over the limit."
            )
        else:
            tail = (f"Median ({med:.0f} km/h) is below the limit — fast-tail, not broad speeding."
                    if pd.notna(med) else "Review road design capacity.")
            primary = (
                f"85th percentile speed (<b>{f85:.0f} km/h</b>) exceeds the posted "
                f"<b>{posted:.0f} km/h</b> limit. {tail}"
            )
        icon, color = "⚡", "#e67e22"

    else:
        primary = (
            f"Speed behaviour broadly matches the posted limit — F85 <b>{f85:.0f} km/h</b>, "
            f"posted <b>{posted:.0f} km/h</b>. Score is driven by road geometry and VRU context."
        )
        icon, color = "ℹ", "#2980b9"

    # Supporting evidence bullets from observed road attributes
    bullets = []
    is_divided = (osm_oneway == "yes") or (osm_lanes >= 4)
    if not is_divided:
        bullets.append("Undivided carriageway — head-on collision exposure at speed")
    else:
        bullets.append("Divided / one-way carriageway — physically separated traffic flows")

    if osm_surface in _UNPAVED:
        bullets.append(f"Unpaved surface ({osm_surface}) — limits safe operating speed")

    cv_lamp = row.get("mapillary_street_lamp", np.nan)
    has_cv_lamp = pd.notna(cv_lamp) and float(cv_lamp) > 0
    if osm_lit == "no":
        bullets.append("No street lighting — elevated risk after dark")
    elif osm_lit == "yes" or has_cv_lamp:
        src = " (Mapillary CV)" if has_cv_lamp and osm_lit != "yes" else ""
        bullets.append(f"Street lighting confirmed{src}")

    if pd.notna(ntl) and float(ntl) > 40:
        bullets.append(f"High nighttime activity (VIIRS: {ntl:.0f}/100) — elevated pedestrian/cyclist exposure after dark")
    elif pd.notna(ntl) and float(ntl) > 15:
        bullets.append(f"Moderate nighttime activity in area (VIIRS: {ntl:.0f}/100)")

    cv_speed    = row.get("mapillary_detected_speed", np.nan)
    cv_mismatch = row.get("mapillary_speed_mismatch", np.nan)
    if pd.notna(cv_speed) and pd.notna(posted):
        if pd.notna(cv_mismatch) and float(cv_mismatch) > 10:
            direction = "below" if float(cv_speed) < float(posted) else "above"
            bullets.append(
                f"Speed sign detected via Mapillary: <b>{cv_speed:.0f} km/h</b> — "
                f"{cv_mismatch:.0f} km/h {direction} posted limit"
            )
        else:
            bullets.append(f"Speed sign detected via Mapillary: {cv_speed:.0f} km/h (consistent with posted)")

    cv_ped = row.get("mapillary_ped_crossing", np.nan)
    if pd.notna(cv_ped) and float(cv_ped) > 0:
        bullets.append(f"Pedestrian crossing(s) detected in area: {cv_ped:.0f} (Mapillary)")

    bullets_html = ""
    if bullets:
        items = "".join(f'<li style="margin:2px 0">{b}</li>' for b in bullets)
        bullets_html = (f'<ul style="margin:5px 0 0 0;padding-left:18px;'
                        f'font-size:11px;color:#444;line-height:1.4">{items}</ul>')

    return (f'<div style="border-left:3px solid {color};background:#fafafa;'
            f'padding:7px 10px;margin:8px 0;border-radius:0 4px 4px 0;'
            f'font-size:12px;line-height:1.5">'
            f'<span style="color:{color}">{icon}</span> {primary}'
            f'{bullets_html}</div>')


def _score_bar(label: str, value, weight_pct: int, color: str = "#666") -> str:
    try:
        w = max(0, min(100, int(float(value)))) if pd.notna(value) else 0
    except (ValueError, TypeError):
        w = 0
    return (f'<div style="margin:4px 0">'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#555">'
            f'<span>{label} <span style="color:#bbb;font-size:10px">({weight_pct}%)</span></span>'
            f'<b style="color:#333">{w}</b></div>'
            f'<div style="background:#e8e8e8;border-radius:3px;height:6px;margin-top:2px">'
            f'<div style="background:{color};width:{w}%;height:6px;border-radius:3px"></div>'
            f'</div></div>')


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
    def _f(col, default=np.nan):
        v = row.get(col, default)
        try:
            return float(v) if pd.notna(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    sss  = _f("sss")
    sl   = _f("speed_limit")
    ss   = _f("ss_limit")
    f85  = _f("speed_85th")
    med  = _f("median_speed")
    band    = row.get("sss_band", "—")
    rc      = row.get("road_class", "—")
    lu      = row.get("land_use", "—")
    ghsl_cls= row.get("ghsl_settlement_class", None)
    country = row.get("country", "—")
    img_url = row.get("image_url", "")

    exposure      = _f("exposure_score")
    dist_school   = _f("dist_to_school_m")
    dist_hospital = _f("dist_to_hospital_m")
    pop_density   = _f("pop_density_500m")
    int_score     = _f("intersection_score")
    pop_component = _f("exposure_component_population")
    tv_component  = _f("exposure_component_traffic")

    # OSM infrastructure tags
    _osm_lanes_raw   = _f("osm_lanes")
    _osm_surface_raw = _stag(row, "osm_surface")
    _osm_lit_raw     = _stag(row, "osm_lit")
    _osm_oneway_raw  = _stag(row, "osm_oneway")
    _osm_junction    = _stag(row, "osm_junction")
    _ntl             = _f("ntl_exposure_score")
    _sinuosity       = _f("sinuosity")

    priority_index = _f("priority_index")
    priority_band  = row.get("priority_band", "—")
    likelihood     = _f("likelihood_score")
    severity       = _f("severity_score")

    align_only      = _f("alignment_only_score")
    align_only_band = row.get("alignment_only_band", "—")

    def fmt(v, unit=""):
        return f"{v:.1f}{unit}" if pd.notna(v) else "—"

    def fmt_dist(v):
        return f"{v:,.0f} m" if pd.notna(v) else "no data"

    header_band  = band if pd.notna(sss) else (align_only_band if pd.notna(align_only) else "—")
    header_color = BAND_COLORS.get(header_band, "#999")
    header_label = f"SSS {fmt(sss)}" if pd.notna(sss) else (
        f"Tier 1: {fmt(align_only)}" if pd.notna(align_only) else "No score")

    # F85 color hint: red if overposted (limit well above F85), orange if underposted
    f85_color = "#333"
    if pd.notna(f85) and pd.notna(sl):
        if sl > f85 + 15:
            f85_color = "#c0392b"
        elif f85 > sl + 15:
            f85_color = "#e67e22"

    # Speed grid — 2×2 layout
    speed_grid = (
        f'<table style="width:100%;border-collapse:separate;border-spacing:3px;margin:8px 0">'
        f'<tr>'
        f'<td style="padding:5px 8px;background:#f5f5f5;border-radius:4px;width:50%">'
        f'<div style="color:#999;font-size:10px;text-transform:uppercase;letter-spacing:0.4px">Posted Limit</div>'
        f'<div style="font-size:18px;font-weight:bold;color:#222">{fmt(sl)}'
        f'<span style="font-size:11px;font-weight:normal;color:#888"> km/h</span></div></td>'
        f'<td style="padding:5px 8px;background:#f5f5f5;border-radius:4px;width:50%">'
        f'<div style="color:#999;font-size:10px;text-transform:uppercase;letter-spacing:0.4px">85th Pct Speed</div>'
        f'<div style="font-size:18px;font-weight:bold;color:{f85_color}">{fmt(f85)}'
        f'<span style="font-size:11px;font-weight:normal;color:#888"> km/h</span></div></td>'
        f'</tr><tr>'
        f'<td style="padding:5px 8px;background:#f9f9f9;border-radius:4px">'
        f'<div style="color:#999;font-size:10px;text-transform:uppercase;letter-spacing:0.4px">Median Speed</div>'
        f'<div style="font-size:14px;font-weight:bold;color:#444">{fmt(med)}'
        f'<span style="font-size:11px;font-weight:normal;color:#888"> km/h</span></div></td>'
        f'<td style="padding:5px 8px;background:#f9f9f9;border-radius:4px">'
        f'<div style="color:#999;font-size:10px;text-transform:uppercase;letter-spacing:0.4px">Safe System Limit</div>'
        f'<div style="font-size:14px;font-weight:bold;color:#444">{fmt(ss)}'
        f'<span style="font-size:11px;font-weight:normal;color:#888"> km/h</span></div></td>'
        f'</tr></table>'
    )

    # Reason block — data-driven, no WHO references
    reason_html = _build_reason_html(row)

    # Context line
    ghsl_label = (ghsl_cls.replace("_", " ").title()
                  if ghsl_cls and str(ghsl_cls) not in ("nan", "None", "no_data") else lu)
    context_line = (f'<div style="font-size:11px;color:#888;margin:2px 0 6px 0">'
                    f'{rc} &nbsp;·&nbsp; {ghsl_label} &nbsp;·&nbsp; {country}</div>')

    # Score breakdown (collapsible)
    score_details = (
        f'<details style="margin:4px 0">'
        f'<summary style="cursor:pointer;color:#1a6fa8;font-size:12px;'
        f'user-select:none;padding:3px 0;list-style:none;outline:none">'
        f'&#9654; Score breakdown</summary>'
        f'<div style="padding:6px 2px 2px 2px">'
        f'{_score_bar("Credibility gap", row.get("sub_score_limit_credibility"), 40, "#e67e22")}'
        f'{_score_bar("VRU context",     row.get("sub_score_vru_risk"),          35, "#1a6fa8")}'
        f'{_score_bar("Limit alignment", row.get("sub_score_limit_alignment"),   25, "#7c3aed")}'
        f'</div></details>'
    )

    # Road infrastructure + nightlights (collapsible)
    def _osm_disp(v):
        return v if v and v not in ("nan", "None", "") else "—"

    lanes_disp   = (f"{int(float(_osm_lanes_raw))}" if pd.notna(_osm_lanes_raw)
                    else "—")
    surface_disp = _osm_disp(_osm_surface_raw)
    lit_disp     = _osm_disp(_osm_lit_raw)
    oneway_disp  = ("one-way" if _osm_oneway_raw == "yes"
                    else "two-way" if _osm_oneway_raw == "no" else "—")
    junction_disp = _osm_disp(_osm_junction)

    ntl_label = "—"
    ntl_bar   = ""
    if pd.notna(_ntl):
        _ntl_f = float(_ntl)
        ntl_label = (f"{_ntl_f:.0f} / 100 "
                     f"({'high' if _ntl_f > 55 else 'moderate' if _ntl_f > 20 else 'low'})")
        ntl_w = max(0, min(100, int(_ntl_f)))
        ntl_bar = (f'<div style="background:#e8e8e8;border-radius:3px;height:5px;margin:2px 0 4px 0">'
                   f'<div style="background:#f59e0b;width:{ntl_w}%;height:5px;border-radius:3px"></div></div>')

    sinuosity_disp = f"{_sinuosity:.2f}" if pd.notna(_sinuosity) else "—"

    infra_details = (
        f'<details style="margin:4px 0">'
        f'<summary style="cursor:pointer;color:#1a6fa8;font-size:12px;'
        f'user-select:none;padding:3px 0;list-style:none;outline:none">'
        f'&#9654; Road infrastructure</summary>'
        f'<div style="padding:6px 2px 2px 2px;font-size:11px;color:#444;line-height:1.8">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr><td style="color:#888;width:45%">Lanes</td><td><b>{lanes_disp}</b></td>'
        f'    <td style="color:#888">Direction</td><td><b>{oneway_disp}</b></td></tr>'
        f'<tr><td style="color:#888">Surface</td><td><b>{surface_disp}</b></td>'
        f'    <td style="color:#888">Lighting</td><td><b>{lit_disp}</b></td></tr>'
        f'<tr><td style="color:#888">Junction</td><td><b>{junction_disp}</b></td>'
        f'    <td style="color:#888">Sinuosity</td><td><b>{sinuosity_disp}</b></td></tr>'
        f'</table>'
        f'<div style="margin-top:6px;border-top:1px solid #eee;padding-top:5px">'
        f'<span style="color:#888">Nighttime activity (VIIRS):</span> <b>{ntl_label}</b>'
        f'{ntl_bar}</div>'
        f'</div></details>'
    )

    # Exposure + Priority (collapsible)
    exp_inner = ""
    if pd.notna(exposure):
        exp_inner = (
            f'<div style="font-size:11px;color:#444;line-height:1.7">'
            f'<b>Exposure: {fmt(exposure)}</b> / 100<br>'
            f'&nbsp;School (12%): {fmt_dist(dist_school)}'
            f'{"  ✓" if pd.notna(dist_school) and dist_school <= 500 else ""}<br>'
            f'&nbsp;Hospital (8%): {fmt_dist(dist_hospital)}'
            f'{"  ✓" if pd.notna(dist_hospital) and dist_hospital <= 750 else ""}<br>'
            f'&nbsp;Population density (25%): {fmt(pop_density)} ppl/km²<br>'
            f'&nbsp;Intersection density (20%): {fmt(int_score)} / 100<br>'
            f'&nbsp;Traffic volume (35%): '
            f'{f"{tv_component:.0f}th pctile" if pd.notna(tv_component) else "—"}'
            f'</div>'
        )
    pi_inner = ""
    if pd.notna(priority_index):
        pi_bc = BAND_COLORS.get(priority_band, "#999")
        pi_inner = (
            f'<div style="margin-top:7px;font-size:11px;color:#444">'
            f'<span style="background:{pi_bc};color:white;padding:2px 6px;'
            f'border-radius:3px;font-size:10px">{priority_band}</span>'
            f'  <b>Priority Index: {fmt(priority_index)}</b><br>'
            f'&nbsp;Likelihood: {fmt(likelihood)} &nbsp;·&nbsp;'
            f' Severity: {fmt(severity)} &nbsp;·&nbsp; Exposure: {fmt(exposure)}'
            f'</div>'
        )
    exp_pi_details = (
        f'<details style="margin:4px 0">'
        f'<summary style="cursor:pointer;color:#1a6fa8;font-size:12px;'
        f'user-select:none;padding:3px 0;list-style:none;outline:none">'
        f'&#9654; Exposure &amp; priority</summary>'
        f'<div style="padding:6px 2px 2px 2px">{exp_inner}{pi_inner}</div>'
        f'</details>'
    )

    # Tier 1 notice
    t1_html = ""
    if pd.isna(sss) and pd.notna(align_only):
        t1_bc = BAND_COLORS.get(align_only_band, "#999")
        t1_html = (
            f'<div style="background:{t1_bc};color:white;padding:4px 8px;'
            f'border-radius:4px;margin:0 0 6px 0;font-size:11px">'
            f'📏 Tier 1 only — no GPS speed data for this segment.<br>'
            f'<span style="font-weight:normal">'
            f'Alignment score: {fmt(align_only)} ({align_only_band})</span></div>'
        )

    img_html = ""
    if img_url and isinstance(img_url, str) and img_url.startswith("http"):
        img_html = (f'<div style="margin-top:7px">'
                    f'<a href="{img_url}" target="_blank" '
                    f'style="font-size:11px;color:#1a6fa8">📷 View street imagery</a>'
                    f'</div>')

    return (
        f'<div style="font-family:Arial,sans-serif;width:360px;font-size:13px">'
        f'<div style="background:{header_color};color:white;padding:8px 12px;'
        f'border-radius:4px 4px 0 0;font-weight:bold;font-size:15px">'
        f'{header_band} &nbsp;·&nbsp; {header_label}</div>'
        f'<div style="padding:10px 12px;border:1px solid #ddd;border-top:none;'
        f'border-radius:0 0 4px 4px;background:#fff">'
        f'{t1_html}'
        f'{speed_grid}'
        f'{reason_html}'
        f'{context_line}'
        f'<hr style="margin:5px 0;border:none;border-top:1px solid #eee">'
        f'{score_details}'
        f'{infra_details}'
        f'{exp_pi_details}'
        f'{img_html}'
        f'</div></div>'
    )


def _build_ml_popup_html(row: pd.Series) -> str:
    """Rich popup for ML-predicted (no GPS) segments — mirrors Tier-1/2 layout."""
    score = row.get("ml_predicted_sss", np.nan)
    band  = row.get("ml_predicted_band", "Moderate")
    img_url = row.get("image_url", "")
    shap_feat = row.get("ml_shap_top_feature", "—")
    rc    = row.get("road_class_norm", row.get("road_class", "—"))
    lu    = row.get("land_use", "—")
    ghsl_cls = row.get("ghsl_settlement_class", None)
    country  = row.get("country", "—")

    # Coerce numerics to plain float to avoid pandas nullable-type formatting issues
    def _f(col, default=np.nan):
        v = row.get(col, default)
        try:
            return float(v) if pd.notna(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    sl   = _f("speed_limit")
    ss   = _f("ss_limit")
    conf = _f("ml_confidence")

    header_color = BAND_COLORS.get(band, "#888")

    def fmt(v, unit=""):
        return f"{float(v):.1f}{unit}" if pd.notna(v) else "—"

    # Speed grid — posted + SS limit; F85/median unavailable (no GPS)
    gap_color = "#333"
    if pd.notna(sl) and pd.notna(ss):
        gap = sl - ss
        if gap > 15:
            gap_color = "#c0392b"
        elif gap < -15:
            gap_color = "#e67e22"

    speed_grid = (
        f'<table style="width:100%;border-collapse:separate;border-spacing:3px;margin:8px 0">'
        f'<tr>'
        f'<td style="padding:5px 8px;background:#f5f5f5;border-radius:4px;width:50%">'
        f'<div style="color:#999;font-size:10px;text-transform:uppercase;letter-spacing:0.4px">Posted Limit</div>'
        f'<div style="font-size:18px;font-weight:bold;color:#222">{fmt(sl)}'
        f'<span style="font-size:11px;font-weight:normal;color:#888"> km/h</span></div></td>'
        f'<td style="padding:5px 8px;background:#f5f5f5;border-radius:4px;width:50%">'
        f'<div style="color:#999;font-size:10px;text-transform:uppercase;letter-spacing:0.4px">Safe System Limit</div>'
        f'<div style="font-size:18px;font-weight:bold;color:{gap_color}">{fmt(ss)}'
        f'<span style="font-size:11px;font-weight:normal;color:#888"> km/h</span></div></td>'
        f'</tr><tr>'
        f'<td style="padding:5px 8px;background:#f9f9f9;border-radius:4px;text-align:center" colspan="2">'
        f'<div style="color:#999;font-size:10px;text-transform:uppercase;letter-spacing:0.4px">85th %ile / Median Speed</div>'
        f'<div style="font-size:13px;color:#aaa;font-style:italic">No GPS data — model estimate</div>'
        f'</td></tr></table>'
    )

    # Speed gap reason
    reason_lines = []
    if pd.notna(sl) and pd.notna(ss):
        gap = sl - ss
        if gap > 15:
            reason_lines.append(f"Posted limit <b>{fmt(sl)} km/h</b> exceeds safe system limit by <b>{gap:.0f} km/h</b> — likely overposted.")
        elif gap < -15:
            reason_lines.append(f"Safe system limit <b>{fmt(ss)} km/h</b> exceeds posted limit — possible underposting.")
        else:
            reason_lines.append(f"Speed limit gap of <b>{abs(gap):.0f} km/h</b> — within tolerable range.")

    cv_lamp = row.get("mapillary_street_lamp", np.nan)
    cv_ped  = row.get("mapillary_ped_crossing", np.nan)
    cv_guard= row.get("mapillary_guardrail", np.nan)
    osm_lit = _stag(row, "osm_lit").lower()
    if pd.notna(cv_ped) and float(cv_ped) > 0:
        reason_lines.append(f"Mapillary: <b>{int(cv_ped)} pedestrian crossing(s)</b> detected nearby — elevated VRU risk.")
    lit_ok = (pd.notna(cv_lamp) and float(cv_lamp) > 0) or (osm_lit in ("yes", "automatic", "24/7"))
    if not lit_ok:
        reason_lines.append("No confirmed street lighting — higher nighttime risk.")
    if pd.notna(cv_guard) and float(cv_guard) > 0:
        reason_lines.append(f"Mapillary: {int(cv_guard)} guardrail segment(s) detected.")

    reason_html = ""
    if reason_lines:
        bullets = "".join(f'<li style="margin-bottom:3px">{r}</li>' for r in reason_lines)
        reason_html = (
            f'<div style="background:#fafafa;border-left:3px solid {header_color};'
            f'padding:7px 10px;border-radius:0 4px 4px 0;margin:4px 0;font-size:12px;color:#333">'
            f'<ul style="margin:0;padding-left:14px;line-height:1.5">{bullets}</ul></div>'
        )

    # Context
    ghsl_label = (ghsl_cls.replace("_", " ").title()
                  if ghsl_cls and str(ghsl_cls) not in ("nan", "None", "no_data") else lu)
    context_line = (f'<div style="font-size:11px;color:#888;margin:2px 0 6px 0">'
                    f'{rc} &nbsp;·&nbsp; {ghsl_label} &nbsp;·&nbsp; {country}</div>')

    # SHAP / model details (collapsible)
    conf_txt = f"{conf:.1f}" if pd.notna(conf) else "—"
    shap_details = (
        f'<details style="margin:4px 0">'
        f'<summary style="cursor:pointer;color:#7c3aed;font-size:12px;'
        f'user-select:none;padding:3px 0;list-style:none;outline:none">'
        f'&#9654; Model details</summary>'
        f'<div style="padding:6px 2px 2px 2px;font-size:11px;color:#444;line-height:1.7">'
        f'<b>Top SHAP driver:</b> {shap_feat}<br>'
        f'<b>Confidence (std):</b> {conf_txt} SSS points<br>'
        f'<span style="color:#aaa">XGBoost trained on Tier-2 scored segments. '
        f'Lower std = higher confidence.</span>'
        f'</div></details>'
    )

    # Exposure (collapsible) if available
    exposure      = row.get("exposure_score", np.nan)
    dist_school   = row.get("dist_to_school_m", np.nan)
    dist_hospital = row.get("dist_to_hospital_m", np.nan)
    pop_density   = row.get("pop_density_500m", np.nan)
    int_score     = row.get("intersection_score", np.nan)
    exp_details = ""
    if any(pd.notna(v) for v in [exposure, dist_school, dist_hospital, pop_density]):
        def fmt_dist(v):
            return f"{v:,.0f} m" if pd.notna(v) else "no data"
        exp_details = (
            f'<details style="margin:4px 0">'
            f'<summary style="cursor:pointer;color:#1a6fa8;font-size:12px;'
            f'user-select:none;padding:3px 0;list-style:none;outline:none">'
            f'&#9654; Exposure context</summary>'
            f'<div style="padding:6px 2px 2px 2px;font-size:11px;color:#444;line-height:1.7">'
            f'{"<b>Exposure: " + fmt(exposure) + "</b> / 100<br>" if pd.notna(exposure) else ""}'
            f'School: {fmt_dist(dist_school)}'
            f'{"  ✓" if pd.notna(dist_school) and dist_school <= 500 else ""}<br>'
            f'Hospital: {fmt_dist(dist_hospital)}'
            f'{"  ✓" if pd.notna(dist_hospital) and dist_hospital <= 750 else ""}<br>'
            f'Population density: {fmt(pop_density)} ppl/km²<br>'
            f'Intersection density: {fmt(int_score)} / 100'
            f'</div></details>'
        )

    # Street view link
    img_html = ""
    if img_url and isinstance(img_url, str) and img_url.startswith("http"):
        img_html = (f'<div style="margin-top:7px">'
                    f'<a href="{img_url}" target="_blank" '
                    f'style="font-size:11px;color:#1a6fa8">📷 View street imagery</a>'
                    f'</div>')

    disclaimer = (
        f'<div style="background:#fff8e1;border:1px solid #f59e0b;padding:5px 8px;'
        f'border-radius:4px;font-size:11px;color:#92400e;margin-top:8px">'
        f'⚠ Model estimate — no GPS speed data. Use for spatial prioritisation only.'
        f'</div>'
    )

    return (
        f'<div style="font-family:Arial,sans-serif;width:360px;font-size:13px">'
        f'<div style="background:{header_color};color:white;padding:8px 12px;'
        f'border-radius:4px 4px 0 0;font-weight:bold;font-size:15px">'
        f'🤖 ML Predicted &nbsp;·&nbsp; {band} ({fmt(score)})</div>'
        f'<div style="padding:10px 12px;border:1px solid #ddd;border-top:none;'
        f'border-radius:0 0 4px 4px;background:#fff">'
        f'{speed_grid}'
        f'{reason_html}'
        f'{context_line}'
        f'<hr style="margin:5px 0;border:none;border-top:1px solid #eee">'
        f'{shap_details}'
        f'{exp_details}'
        f'{img_html}'
        f'{disclaimer}'
        f'</div></div>'
    )


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
    # CartoDB Positron — very light/minimal base, maximises contrast with
    # our coloured road lines. Set as default (first TileLayer added).
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    m.get_root().html.add_child(folium.Element(_build_legend_html()))
    m.get_root().html.add_child(folium.Element(_build_summary_html(scored, gdf, tier1_only)))
    m.get_root().html.add_child(folium.Element(_build_methodology_html()))

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
            weight = (5 if band == "Critical"
                      else 4 if band == "High Risk"
                      else 3)

            popup_html  = _build_popup_html(row)
            tooltip_txt = f"{band} | SSS: {sss:.1f}" if pd.notna(sss) else "No data"

            try:
                if geom.geom_type in ("LineString", "MultiLineString"):
                    for seg_coords in _geom_to_latlon_list(geom):
                        folium.PolyLine(
                            locations=seg_coords, color=color, weight=weight,
                            opacity=0.9,
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

    # ── Schools, Hospitals, Population, Intersections ───────────────────
    # Loaded directly from the same enrichment_data/ files the Exposure
    # score already uses (see enrichment.py) — these were previously only
    # a number buried in the popup; now they're visible on the map itself.
    schools_gdf = _load_amenities(f"{data_dir}/schools", "Schools (map)")
    if len(schools_gdf):
        plot_schools = schools_gdf
        if len(schools_gdf) > max_amenity_markers:
            plot_schools = schools_gdf.sample(max_amenity_markers, random_state=42)
            print(f"  Sampling {max_amenity_markers:,} of {len(schools_gdf):,} schools for map performance")
        fg_schools = folium.FeatureGroup(
            name=f"🏫 Schools ({len(schools_gdf):,}, {SCHOOL_BUFFER_M:.0f}m buffer)", show=True)
        cluster_schools = MarkerCluster(name="schools_cluster").add_to(fg_schools)
        for _, row in plot_schools.iterrows():
            try:
                c = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
                folium.CircleMarker(
                    location=[c.y, c.x], radius=4, color="#2563eb",
                    fill=True, fill_color="#2563eb", fill_opacity=0.9,
                    tooltip="School",
                ).add_to(cluster_schools)
            except Exception:
                pass
        fg_schools.add_to(m)

    hosp_gdf = _load_amenities(f"{data_dir}/hospitals", "Hospitals (map)")
    if len(hosp_gdf):
        plot_hosp = hosp_gdf
        if len(hosp_gdf) > max_amenity_markers:
            plot_hosp = hosp_gdf.sample(max_amenity_markers, random_state=42)
            print(f"  Sampling {max_amenity_markers:,} of {len(hosp_gdf):,} hospitals for map performance")
        fg_hosp = folium.FeatureGroup(
            name=f"🏥 Hospitals ({len(hosp_gdf):,}, {HOSPITAL_BUFFER_M:.0f}m buffer)", show=True)
        cluster_hosp = MarkerCluster(name="hospitals_cluster").add_to(fg_hosp)
        for _, row in plot_hosp.iterrows():
            try:
                c = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
                folium.CircleMarker(
                    location=[c.y, c.x], radius=4, color="#dc2626",
                    fill=True, fill_color="#dc2626", fill_opacity=0.9,
                    tooltip="Hospital",
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

    # ML-predicted layer (off by default — these are model estimates, not
    # measured values; dashed style distinguishes them from scored segments)
    if "ml_predicted_sss" in gdf.columns:
        ml_segs = gdf[gdf["ml_predicted_sss"].notna()].copy()
        if len(ml_segs):
            if len(ml_segs) > max_segments:
                ml_segs = ml_segs.sample(max_segments, random_state=42)
            fg_ml = folium.FeatureGroup(
                name="ML Predicted (unscored, off by default)", show=False
            )
            from config import BAND_COLORS
            for _, row in ml_segs.iterrows():
                coords = _geom_to_latlon_list(row.geometry)
                if not coords:
                    continue
                band  = row.get("ml_predicted_band", "Moderate")
                color = BAND_COLORS.get(band, "#888888")
                score = row.get("ml_predicted_sss", float("nan"))
                popup_html = _build_ml_popup_html(row)
                score_str = f"{score:.0f}" if pd.notna(score) else "—"
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=2,
                    opacity=0.6,
                    dash_array="6 4",
                    popup=folium.Popup(popup_html, max_width=380),
                    tooltip=f"🤖 ML: {score_str} ({band})",
                ).add_to(fg_ml)
            fg_ml.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
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
        f'<div style="display:flex;align-items:center;margin-bottom:4px">'
        f'<div style="width:16px;height:16px;background:{color};border-radius:2px;margin-right:8px"></div>'
        f'<span>{band}</span></div>'
        for band, color in BAND_COLORS.items()
    )
    return f"""
    <div id="legend-panel" style="position:fixed;bottom:40px;left:12px;z-index:9999;
                background:rgba(30,30,30,0.92);color:white;
                border-radius:8px;font-family:Arial,sans-serif;
                font-size:13px;box-shadow:0 2px 12px rgba(0,0,0,0.4)">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  padding:10px 14px 10px 18px;cursor:pointer"
           onclick="(function(){{
             var b=document.getElementById('legend-body');
             var t=document.getElementById('legend-toggle');
             if(b.style.display==='none'){{b.style.display='block';t.textContent='⊟';}}
             else{{b.style.display='none';t.textContent='⊞';}}
           }})()">
        <b style="font-size:14px">Speed Safety Score</b>
        <span id="legend-toggle" style="margin-left:14px;font-size:16px;line-height:1">⊟</span>
      </div>
      <div id="legend-body" style="padding:0 18px 14px 18px">
        <hr style="border-color:#555;margin:0 0 8px 0">
        {items}
        <div style="margin-top:8px;display:flex;align-items:center">
          <div style="width:12px;height:12px;background:#2563eb;border-radius:50%;margin-right:8px"></div>
          <span style="font-size:12px">Schools</span>
        </div>
        <div style="margin-top:4px;display:flex;align-items:center">
          <div style="width:12px;height:12px;background:#dc2626;border-radius:50%;margin-right:8px"></div>
          <span style="font-size:12px">Hospitals</span>
        </div>
        <div style="margin-top:8px;display:flex;align-items:center">
          <div style="width:12px;height:3px;background:#7c3aed;margin-right:8px;
                      border-top:2px dashed #7c3aed"></div>
          <span style="font-size:12px">Blindspot (high SSS + no imagery)</span>
        </div>
        <div style="color:#aaa;font-size:11px;margin-top:8px">
          Same color scale for SSS and Priority Index layers
        </div>
        <div style="color:#aaa;font-size:11px;margin-top:6px">AI for Safer Roads · ADB Challenge</div>
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
                max-width:300px;background:rgba(30,30,30,0.92);color:white;
                border-radius:8px;font-family:Arial,sans-serif;
                font-size:12px;box-shadow:0 2px 12px rgba(0,0,0,0.4)">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  padding:10px 12px 10px 16px;cursor:pointer"
           onclick="(function(){{
             var b=document.getElementById('methodology-body');
             var t=document.getElementById('methodology-toggle');
             if(b.style.display==='none'){{b.style.display='block';t.textContent='⊟';}}
             else{{b.style.display='none';t.textContent='⊞';}}
           }})()">
        <b style="font-size:13px">How Exposure is calculated</b>
        <span id="methodology-toggle" style="margin-left:14px;font-size:16px;line-height:1">⊟</span>
      </div>
      <div id="methodology-body" style="padding:0 16px 12px 16px">
        <hr style="border-color:#555;margin:0 0 8px 0">
        <div style="color:#ccc;line-height:1.5">
          <b>Schools (12%):</b> distance from road to nearest school point
          (HOTOSM data), decaying to 0 at 2× the 500m reference buffer.<br>
          <b>Hospitals (8%):</b> same decay, 750m reference buffer.<br>
          <b>Population (25%):</b> WorldPop density sampled along a 500m
          road buffer, then percentile-ranked within country.<br>
          <b>Intersections (20%):</b> OSM junction count within a 1km
          buffer, per km of road.<br>
          <b>Traffic volume (35%):</b> GPS probe sample count
          (WeightedSample), percentile-ranked within country.
        </div>
        <div style="color:#888;font-size:10px;margin-top:8px">
          Click any road segment for that segment's actual input values.
          Weights with no data in a given run are dropped and the remaining
          weights are redistributed proportionally (see enrichment.py).
        </div>
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
            ml_row = (
                f'<br><b style="color:#c4b5fd">🤖 ML Coverage Extension:</b> {n_ml:,} unscored segments '
                f'predicted (toggle layer below)<br>'
                f'<span style="color:#aaa;font-size:11px">XGBoost trained on Tier 2 labels — '
                f'triage only, not for enforcement</span>'
            )

    return f"""
    <div id="summary-panel" style="position:fixed;top:52px;left:12px;z-index:9990;
                background:rgba(30,30,30,0.92);color:white;
                padding:10px 14px;border-radius:8px;font-family:Arial,sans-serif;
                font-size:12px;box-shadow:0 2px 12px rgba(0,0,0,0.4);min-width:220px;
                max-height:80vh;overflow-y:auto">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:4px;cursor:pointer"
           onclick="(function(){{
             var b=document.getElementById('summary-body');
             var t=document.getElementById('summary-toggle');
             if(b.style.display==='none'){{b.style.display='block';t.textContent='⊟';}}
             else{{b.style.display='none';t.textContent='⊞';}}
           }})()">
        <b style="font-size:13px">Analysis Summary</b>
        <span id="summary-toggle" style="font-size:16px;line-height:1;padding:0 4px">⊟</span>
      </div>
      <div id="summary-body">
      <hr style="border-color:#555;margin:4px 0">
      <b>Total scored segments (Tier 2, full SSS):</b> {total:,}<br><br>
      <table style="width:100%;border-collapse:collapse">
        <tr style="color:#aaa"><td>Band</td><td style="text-align:right">N</td><td style="text-align:right">%</td></tr>
        {band_rows}
      </table>
      <br>
      <table style="width:100%;border-collapse:collapse">
        <tr style="color:#aaa"><td>Country</td><td style="text-align:right">Avg SSS</td>
            <td style="text-align:right">Scored</td><td style="text-align:right">Coverage</td></tr>
        {coverage_rows}
      </table>
      <div style="color:#f59e0b;font-size:10px;margin-top:6px">
        ⚠ Coverage = % of total segments with speed data.<br>
        Unscored segments are not classified.
      </div>
      {tier1_row}
      {pi_row}
      {ml_row}
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
    explainer  = shap_lib.TreeExplainer(model)
    sv         = np.abs(explainer(X).values)
    mean_shap  = sv.mean(axis=0)

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
