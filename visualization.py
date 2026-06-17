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
    sss      = row.get("sss", np.nan)
    band     = row.get("sss_band", "—")
    sl       = row.get("speed_limit", np.nan)
    ss       = row.get("ss_limit", np.nan)
    f85      = row.get("speed_85th", np.nan)
    med      = row.get("median_speed", np.nan)
    pct_over = row.get("pct_over_limit", np.nan)
    rc       = row.get("road_class", "—")
    lu       = row.get("land_use", "—")
    country  = row.get("country", "—")
    rec      = row.get("sss_recommendation", "")
    img_url  = row.get("image_url", "")
    seg_id   = row.get("segment_id", "—")

    # Exposure — now broken into its actual calculation inputs, not a
    # single combined number, per the request to show HOW it's computed.
    exposure       = row.get("exposure_score", np.nan)
    dist_school    = row.get("dist_to_school_m", np.nan)
    dist_hospital  = row.get("dist_to_hospital_m", np.nan)
    pop_density    = row.get("pop_density_500m", np.nan)
    int_score      = row.get("intersection_score", np.nan)
    pop_component  = row.get("exposure_component_population", np.nan)
    tv_component   = row.get("exposure_component_traffic", np.nan)

    # Priority Index fields (runs alongside SSS — see priority_scoring.py)
    priority_index = row.get("priority_index", np.nan)
    priority_band  = row.get("priority_band", "—")
    likelihood     = row.get("likelihood_score", np.nan)
    severity       = row.get("severity_score", np.nan)

    # Tier 1 alignment-only score (covers segments with no behavioural data)
    align_only      = row.get("alignment_only_score", np.nan)
    align_only_band = row.get("alignment_only_band", "—")

    band_color = BAND_COLORS.get(band, "#999")

    img_html = ""
    if img_url and isinstance(img_url, str) and img_url.startswith("http"):
        img_html = f'<a href="{img_url}" target="_blank">📷 View street imagery</a><br>'

    def fmt(v, unit=""):
        return f"{v:.1f}{unit}" if pd.notna(v) else "—"

    def fmt_dist(v):
        if pd.isna(v):
            return "none within 50km (or no data loaded for this area)"
        return f"{v:,.0f}m"

    # Exposure breakdown badge — every input visible with its weight and
    # what it's actually measuring, instead of one opaque number.
    exp_html = ""
    if pd.notna(exposure):
        exp_html = f"""
        <div style="background:#1a6fa8;color:white;padding:6px 8px;
                    border-radius:4px;margin:4px 0;font-size:11px">
          👥 <b>Exposure: {fmt(exposure)}</b> / 100<br>
          <span style="font-weight:normal">
          &nbsp;Nearest school (12% wt): {fmt_dist(dist_school)} {'(within 500m buffer)' if pd.notna(dist_school) and dist_school <= 500 else ''}<br>
          &nbsp;Nearest hospital (8% wt): {fmt_dist(dist_hospital)} {'(within 750m buffer)' if pd.notna(dist_hospital) and dist_hospital <= 750 else ''}<br>
          &nbsp;Population, 500m buffer (25% wt): {fmt(pop_density)} ppl/km²{f' ({pop_component:.0f}th pctile)' if pd.notna(pop_component) and pd.notna(pop_density) else ''}<br>
          &nbsp;Intersection density (20% wt): {fmt(int_score)}/100<br>
          &nbsp;Traffic volume (35% wt): {f'{tv_component:.0f}th percentile (country)' if pd.notna(tv_component) else '—'}
          </span>
        </div>"""

    # Priority Index badge — SECONDARY "where to act first" layer, shown
    # alongside SSS (not replacing it).
    pi_html = ""
    if pd.notna(priority_index):
        pi_band_color = BAND_COLORS.get(priority_band, "#999")
        pi_html = f"""
        <div style="background:{pi_band_color};color:white;padding:4px 8px;
                    border-radius:4px;margin:4px 0;font-size:11px">
          🎯 Priority Index (secondary): {fmt(priority_index)} ({priority_band}) &nbsp;|&nbsp;
          L: {fmt(likelihood)} · S: {fmt(severity)} · E: {fmt(exposure)}
        </div>"""

    # Tier 1 badge — only shown for segments that have alignment-only data
    # (no F85/median), so it's clear why SSS itself might be blank.
    t1_html = ""
    if pd.isna(sss) and pd.notna(align_only):
        t1_band_color = BAND_COLORS.get(align_only_band, "#999")
        t1_html = f"""
        <div style="background:{t1_band_color};color:white;padding:4px 8px;
                    border-radius:4px;margin:4px 0;font-size:11px">
          📏 Tier 1 only — posted limit vs Safe System standard: {fmt(align_only)} ({align_only_band})<br>
          <span style="font-weight:normal">No GPS behavioural data (F85/median) for this segment — full SSS unavailable.</span>
        </div>"""

    header_band  = band if pd.notna(sss) else (align_only_band if pd.notna(align_only) else "—")
    header_color = BAND_COLORS.get(header_band, "#999")
    header_label = f"SSS: {fmt(sss)}" if pd.notna(sss) else (
        f"Tier 1 only: {fmt(align_only)}" if pd.notna(align_only) else "No score")

    return f"""
    <div style="font-family:Arial,sans-serif;width:360px;font-size:13px">
      <div style="background:{header_color};color:white;padding:8px 12px;
                  border-radius:4px 4px 0 0;font-weight:bold;font-size:15px">
        {header_band} &nbsp;·&nbsp; {header_label}
      </div>
      <div style="padding:10px 12px;border:1px solid #ddd;border-top:none;border-radius:0 0 4px 4px">
        {t1_html}
        {exp_html}
        {pi_html}
        <b>Segment:</b> {seg_id}<br>
        <b>Country:</b> {country}<br>
        <b>Road class:</b> {rc}<br>
        <b>Land use:</b> {lu}<br>
        <hr style="margin:6px 0">
        <b>Posted limit:</b> {fmt(sl, ' km/h')}<br>
        <b>Safe System limit:</b> {fmt(ss, ' km/h')}<br>
        <b>Median speed:</b> {fmt(med, ' km/h')}<br>
        <b>85th pct speed:</b> {fmt(f85, ' km/h')}<br>
        <b>% over limit:</b> {fmt(pct_over, '%')}<br>
        <hr style="margin:6px 0">
        <b>Sub-scores:</b><br>
        &nbsp;Limit alignment: {fmt(row.get('sub_score_limit_alignment'))}<br>
        &nbsp;Limit credibility gap: {fmt(row.get('sub_score_limit_credibility'))}<br>
        &nbsp;VRU risk: {fmt(row.get('sub_score_vru_risk'))}<br>
        &nbsp;Compliance (context only, not scored): {fmt(row.get('sub_score_compliance'))}<br>
        {f'''<hr style="margin:6px 0">
        <b>Priority Index sub-scores:</b><br>
        &nbsp;Likelihood: limit credibility {fmt(row.get("sub_likelihood_speed_gap"))},
          credibility {fmt(row.get("sub_likelihood_credibility"))},
          variability {fmt(row.get("sub_likelihood_variability"))}<br>
        &nbsp;Severity: Safe System {fmt(row.get("sub_severity_safe_system"))},
          Nilsson {fmt(row.get("sub_severity_nilsson"))},
          infrastructure {fmt(row.get("sub_severity_infrastructure"))},
          helmet {fmt(row.get("sub_severity_helmet"))}<br>''' if pd.notna(priority_index) else ''}
        <hr style="margin:6px 0">
        <i style="font-size:11px;color:#555">{rec}</i><br>
        {img_html}
      </div>
    </div>
    """


def build_interactive_map(
    gdf: gpd.GeoDataFrame,
    corridors: gpd.GeoDataFrame = None,
    output_path: str = "speed_safety_map.html",
    max_segments: int = 5000,
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
    m.get_root().html.add_child(folium.Element(_build_methodology_html()))

    # Segment layers per country
    for country_code in scored["country_code"].unique():
        country_sub  = scored[scored["country_code"] == country_code]
        country_name = country_sub["country"].iloc[0]
        fg = folium.FeatureGroup(name=f"📍 {country_name}", show=True)

        plot_sub = country_sub
        if len(country_sub) > max_segments:
            plot_sub = country_sub.sample(max_segments, random_state=42)
            print(f"  Sampling {max_segments:,} of {len(country_sub):,} {country_code} segments for map performance")

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

    folium.LayerControl(collapsed=False).add_to(m)
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
        <span style="font-size:12px">Schools</span>
      </div>
      <div style="margin-top:4px;display:flex;align-items:center">
        <div style="width:12px;height:12px;background:#dc2626;border-radius:50%;margin-right:8px"></div>
        <span style="font-size:12px">Hospitals</span>
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
                max-width:300px;background:rgba(30,30,30,0.92);color:white;
                padding:12px 16px;border-radius:8px;font-family:Arial,sans-serif;
                font-size:12px;box-shadow:0 2px 12px rgba(0,0,0,0.4)">
      <b style="font-size:13px">How Exposure is calculated</b>
      <hr style="border-color:#555;margin:6px 0">
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

    return f"""
    <div style="position:fixed;top:12px;right:12px;z-index:9999;
                background:rgba(30,30,30,0.92);color:white;
                padding:14px 18px;border-radius:8px;font-family:Arial,sans-serif;
                font-size:12px;box-shadow:0 2px 12px rgba(0,0,0,0.4);min-width:240px">
      <b style="font-size:14px">Analysis Summary</b>
      <hr style="border-color:#555;margin:6px 0">
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
