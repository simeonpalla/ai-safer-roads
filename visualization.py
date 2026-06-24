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
    sss           = row.get("sss", np.nan)
    band          = row.get("sss_band", "—")
    sl            = row.get("speed_limit", np.nan)
    ss            = row.get("ss_limit", np.nan)
    f85           = row.get("speed_85th", np.nan)
    med           = row.get("median_speed", np.nan)
    pct_over      = row.get("pct_over_limit", np.nan)
    rc            = row.get("road_class_norm", row.get("road_class", "—"))
    lu            = row.get("land_use", "—")
    country       = row.get("country", row.get("country_code", "—"))
    cc            = row.get("country_code", "")
    seg_id        = row.get("segment_id", "—")
    rec           = row.get("sss_recommendation", "")
    img_url       = row.get("image_url", "")
    credibility   = row.get("credibility_class", "")
    nilsson       = row.get("nilsson_fatal_ratio", np.nan)
    sinuosity     = row.get("sinuosity", np.nan)
    priority_index = row.get("priority_index", np.nan)
    priority_band  = row.get("priority_band", "—")
    spread        = (f85 - med) if pd.notna(f85) and pd.notna(med) else np.nan

    band_color = BAND_COLORS.get(band, "#999")

    def fmt(v, unit="", dec=1):
        return f"{v:.{dec}f}{unit}" if pd.notna(v) else "—"

    # ── Q1: Is the posted limit right? ───────────────────────────────────────
    if pd.notna(sl) and pd.notna(ss):
        gap = sl - ss
        if sl > ss + 5:
            q1_verdict   = "✗ TOO HIGH"
            q1_color     = "#c0392b"
            q1_detail    = f"Posted {sl:.0f} km/h — Safe System standard is {ss:.0f} km/h (+{gap:.0f} km/h over)"
        elif sl < ss * 0.80:
            q1_verdict   = "✗ TOO LOW (outdated / non-credible)"
            q1_color     = "#e67e22"
            q1_detail    = f"Posted {sl:.0f} km/h — Safe System standard is {ss:.0f} km/h (limit {abs(gap):.0f} km/h below road design speed)"
        elif sl < ss - 5:
            q1_verdict   = "⚠ SLIGHTLY LOW"
            q1_color     = "#f39c12"
            q1_detail    = f"Posted {sl:.0f} km/h vs Safe System {ss:.0f} km/h — may be overly restrictive"
        else:
            q1_verdict   = "✓ APPROPRIATE"
            q1_color     = "#27ae60"
            q1_detail    = f"Posted {sl:.0f} km/h aligns with Safe System standard ({ss:.0f} km/h)"
    else:
        q1_verdict = "— NO DATA"
        q1_color   = "#7f8c8d"
        q1_detail  = "No posted speed limit recorded for this segment"

    # ── Q2: Why? ─────────────────────────────────────────────────────────────
    reasons = []
    if pd.notna(sl) and pd.notna(ss) and sl > ss + 5:
        reasons.append(f"Limit {sl:.0f} km/h exceeds {ss:.0f} km/h Safe System ceiling for {lu} {rc} road")
    if pd.notna(sl) and pd.notna(ss) and sl < ss * 0.80:
        reasons.append(f"Limit {sl:.0f} km/h is {ss-sl:.0f} km/h below design speed — likely outdated or never revised after road upgrade")
    if pd.notna(f85) and pd.notna(sl) and f85 > sl + 15:
        if pd.notna(med) and med > sl:
            reasons.append(f"Typical driver (median {med:.0f} km/h) AND fast tail (F85 {f85:.0f} km/h) both exceed posted limit — systemic non-compliance")
        else:
            reasons.append(f"F85 {f85:.0f} km/h exceeds limit by {f85-sl:.0f} km/h — fast-tail behaviour (median {fmt(med,' km/h')} within limit)")
    if pd.notna(spread) and spread > 20:
        reasons.append(f"Wide speed distribution (F85−median = {spread:.0f} km/h) — heavy mixed traffic, GPS data is car-biased")
    if credibility == "Non-Credible":
        reasons.append("Limit non-credible — F85 >20 km/h above posted, drivers ignore signage")
    if credibility == "Infrastructure-Forced":
        reasons.append("Low speeds likely from speed bumps/tables, not genuine compliance — investigate infrastructure")
    if credibility == "Under-Speed":
        reasons.append(f"F85 well below posted limit — road conditions, surface, or geometry forcing lower speeds")
    if pd.notna(nilsson) and nilsson > 4:
        reasons.append(f"Fatal crash risk {nilsson:.1f}× Safe System baseline (Nilsson Power Model)")
    if pd.notna(sinuosity) and sinuosity >= 1.50:
        reasons.append(f"Sharply curved alignment (SI={sinuosity:.2f}) — sight distance limits safe speed")
    if cc == "TH" and rc in ("secondary", "tertiary", "primary"):
        reasons.append("PTW high-risk corridor — Thailand PTW riders account for 74% of road fatalities (WHO 2023)")
    if cc == "MH" and rc in ("primary", "trunk"):
        reasons.append("PTW-truck conflict zone — Maharashtra PTW 37% of fatalities; undivided carriageway (iRAP 1-2 star)")
    if not reasons:
        reasons.append("Limit broadly appropriate; no major credibility or alignment issues detected")

    reasons_html = "".join(
        f'<div style="margin:3px 0;padding:3px 6px;background:#f8f9fa;'
        f'border-left:3px solid #c0392b;border-radius:2px;font-size:12px">'
        f'• {r}</div>' for r in reasons
    )

    # ── Q3: Intervention ─────────────────────────────────────────────────────
    actions = []
    if pd.notna(sl) and pd.notna(ss) and sl > ss + 5:
        rec_limit = row.get("recommended_limit", ss)
        reduction = sl - (rec_limit if pd.notna(rec_limit) else ss)
        effort    = row.get("change_effort", "")
        actions.append(("🚦", f"Reduce speed limit to {rec_limit:.0f} km/h (−{reduction:.0f} km/h) [{effort}]"))
    if pd.notna(sl) and pd.notna(ss) and sl < ss * 0.80:
        actions.append(("🚦", f"Review posted limit ({sl:.0f} km/h) — likely outdated vs Safe System standard ({ss:.0f} km/h). Commission road audit before raising limit."))
    if credibility == "Non-Credible":
        actions.append(("📋", "Redesign limit scheme — signage widely ignored. Physical calming or enforcement cameras required."))
    if credibility == "Infrastructure-Forced":
        actions.append(("🔍", "Verify speed bump/table presence on-site before recommending any limit change."))
    if pd.notna(nilsson) and nilsson > 4 and rc in ("trunk", "primary"):
        actions.append(("🛡️", "Install median barrier / physical separation — fatal crash risk >4× baseline on undivided highway"))
    if pd.notna(sinuosity) and sinuosity >= 1.50:
        actions.append(("⚠️", "Install curve warning chevrons and advance warning signs (SI ≥1.50)"))
    elif pd.notna(sinuosity) and sinuosity >= 1.20:
        actions.append(("⚠️", "Install curve advisory speed signs (SI ≥1.20)"))
    if cc == "TH" and rc in ("secondary", "tertiary", "primary"):
        actions.append(("🏍️", "Deploy PTW-targeted enforcement / awareness campaign"))
    if cc == "MH" and rc in ("primary", "trunk"):
        actions.append(("🏍️", "PTW-truck conflict: install rumble strips / hard shoulder demarcation"))
    if not actions:
        actions.append(("📊", "Monitor — schedule next audit in 12 months"))

    # Responsible authority
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
    authority = (authority_map_mh if "Maharashtra" in country or cc == "MH" else authority_map_th).get(rc, "Road Authority")

    actions_html = "".join(
        f'<div style="margin:3px 0;padding:4px 6px;background:#eaf4fb;'
        f'border-left:3px solid #2980b9;border-radius:2px;font-size:12px">'
        f'{icon} {a}</div>' for icon, a in actions
    )

    img_html = ""
    if img_url and isinstance(img_url, str) and img_url.startswith("http"):
        img_html = f'<div style="margin-top:8px">📷 <a href="{img_url}" target="_blank" style="color:#2980b9">View street imagery</a></div>'

    # Technical detail section (collapsed by default)
    tech_id = f"tech_{seg_id}".replace("/", "_").replace(" ", "_")
    align_score = row.get("sub_score_limit_alignment", np.nan)
    cred_score  = row.get("sub_score_limit_credibility", np.nan)
    vru_score   = row.get("sub_score_vru_risk", np.nan)
    nilsson_low  = row.get("nilsson_fatal_ratio_low", np.nan)
    nilsson_high = row.get("nilsson_fatal_ratio_high", np.nan)
    pi_html = ""
    if pd.notna(priority_index):
        pi_color = BAND_COLORS.get(priority_band, "#999")
        pi_html = f'<div style="margin-top:4px;padding:3px 6px;background:{pi_color}20;border-radius:3px;font-size:11px;color:#333">🎯 Priority Index: <b>{priority_index:.1f}</b> ({priority_band})</div>'

    nilsson_range = ""
    if pd.notna(nilsson_low) and pd.notna(nilsson_high):
        nilsson_range = f" <span style='color:#777'>[{nilsson_low:.1f}–{nilsson_high:.1f} Asian range]</span>"

    tech_html = f"""
    <div style="margin-top:8px">
      <div onclick="var d=document.getElementById('{tech_id}');d.style.display=d.style.display=='none'?'block':'none'"
           style="cursor:pointer;color:#2980b9;font-size:11px;user-select:none">
        ▶ Technical detail (sub-scores, Nilsson, credibility)
      </div>
      <div id="{tech_id}" style="display:none;margin-top:6px;font-size:11px;color:#555;line-height:1.6">
        <b>Segment:</b> {seg_id} &nbsp;|&nbsp; <b>Country:</b> {country}<br>
        <b>Median speed:</b> {fmt(med,' km/h')} &nbsp;|&nbsp; <b>F85:</b> {fmt(f85,' km/h')}<br>
        <b>% over limit:</b> {fmt(pct_over,'%')} (context only — not scored)<br>
        <hr style="margin:4px 0;border-color:#eee">
        <b>SSS sub-scores:</b><br>
        &nbsp;Alignment (posted vs Safe System): {fmt(align_score)}/100<br>
        &nbsp;Credibility gap (dual-signal F85+median): {fmt(cred_score)}/100<br>
        &nbsp;VRU context risk (with PTW weighting): {fmt(vru_score)}/100<br>
        <b>Nilsson fatal risk ratio:</b> {fmt(nilsson,'×')}{nilsson_range}<br>
        <b>Credibility class:</b> {credibility if credibility else '—'}<br>
        <b>Sinuosity index:</b> {fmt(sinuosity)}<br>
        {pi_html}
      </div>
    </div>"""

    # Speed facts line — shown in Q1 box so reader sees context immediately
    spread_txt = f" · spread {spread:.0f} km/h" if pd.notna(spread) else ""
    speed_facts = ""
    if pd.notna(med) or pd.notna(f85):
        speed_facts = (
            f'<div style="margin-top:4px;font-size:11px;color:#555;'
            f'background:#f0f0f0;padding:3px 6px;border-radius:3px">'
            f'📊 Median {fmt(med," km/h")} · F85 {fmt(f85," km/h")}{spread_txt}'
            f'</div>'
        )

    return f"""
    <div style="font-family:Arial,sans-serif;width:380px;font-size:13px;line-height:1.4">
      <div style="background:{band_color};color:white;padding:8px 12px;
                  border-radius:4px 4px 0 0;font-weight:bold;font-size:14px">
        {band} &nbsp;·&nbsp; SSS {fmt(sss)}/100 &nbsp;·&nbsp; {lu} {rc}
      </div>
      <div style="padding:10px 12px;border:1px solid #ddd;border-top:none;border-radius:0 0 4px 4px">

        <div style="margin-bottom:8px;padding:6px 8px;background:{q1_color}18;
                    border:1px solid {q1_color}44;border-radius:4px">
          <div style="font-weight:bold;color:{q1_color};font-size:13px">
            ❶ IS THE LIMIT RIGHT? &nbsp; {q1_verdict}
          </div>
          <div style="color:#333;margin-top:2px;font-size:12px">{q1_detail}</div>
          {speed_facts}
        </div>

        <div style="margin-bottom:8px;padding:6px 8px;background:#fdf2f2;
                    border:1px solid #e8c5c5;border-radius:4px">
          <div style="font-weight:bold;color:#7f1d1d;font-size:13px">❷ WHY?</div>
          <div style="margin-top:4px">{reasons_html}</div>
        </div>

        <div style="margin-bottom:6px;padding:6px 8px;background:#eaf4fb;
                    border:1px solid #b3d9f2;border-radius:4px">
          <div style="font-weight:bold;color:#1a4f72;font-size:13px">❸ INTERVENTION</div>
          <div style="margin-top:4px">{actions_html}</div>
          <div style="margin-top:6px;font-size:11px;color:#555">
            <b>Responsible authority:</b> {authority}
          </div>
        </div>

        {tech_html}
        {img_html}
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
        f'<div style="display:flex;align-items:center;margin-bottom:4px">'
        f'<div style="width:16px;height:16px;background:{color};border-radius:2px;margin-right:8px"></div>'
        f'<span>{band}</span></div>'
        for band, color in BAND_COLORS.items()
    )
    return f"""
    <div style="position:fixed;bottom:40px;left:12px;z-index:9999;
                background:rgba(30,30,30,0.92);color:white;
                padding:14px 18px;border-radius:8px;font-family:Arial,sans-serif;
                font-size:13px;box-shadow:0 2px 12px rgba(0,0,0,0.4)">
      <b style="font-size:14px">Speed Safety Score</b><br>
      <hr style="border-color:#555;margin:6px 0">
      {items}
      <div style="margin-top:8px;display:flex;align-items:center">
        <div style="width:12px;height:12px;background:#2563eb;border-radius:50%;margin-right:8px"></div>
        <span style="font-size:12px">Schools (toggle layer — near roads only)</span>
      </div>
      <div style="margin-top:4px;display:flex;align-items:center">
        <div style="width:12px;height:12px;background:#dc2626;border-radius:50%;margin-right:8px"></div>
        <span style="font-size:12px">Hospitals (toggle layer — near roads only)</span>
      </div>
      <div style="margin-top:8px;display:flex;align-items:center">
        <div style="width:12px;height:3px;background:#7c3aed;margin-right:8px;
                    border-top:2px dashed #7c3aed"></div>
        <span style="font-size:12px">Blindspot (high SSS + no imagery)</span>
      </div>
      <div style="color:#aaa;font-size:11px;margin-top:8px">
        Same color scale used for both SSS (default view) and Priority Index<br>
        (toggle "🎯 Priority Index View" in layer control to compare)
      </div>
      <div style="color:#aaa;font-size:11px;margin-top:6px">AI for Safer Roads · ADB Challenge</div>
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
