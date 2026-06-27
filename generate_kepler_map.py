"""
generate_kepler_map.py — Build a full interactive Speed Safety Score map
showing ALL scored segments (no 500-per-country limit).

Unlike speed_safety_map.html (Folium, 1,000 segments max due to popup HTML size),
this map uses Leaflet with WebGL canvas rendering and loads all 14,711 GPS-confirmed
segments. All data is embedded — no server required. Works in any browser.

Features:
  - All 14,711 scored segments as colour-coded dots (Critical/High Risk/Moderate/Acceptable)
  - Band toggle buttons with live counts
  - Click any dot → split-panel detail opens on the right
  - Map auto-pans to the selected segment
  - ❶ IS THE LIMIT RIGHT? verdict block (green/amber/red)
  - ❷ WHY? — reason cards from GPS evidence
  - ❸ INTERVENTIONS — action cards with Urgent/Priority/Recommended badges
  - Score Metrics tab — sub-score bars, SSS formula, GPS evidence grid
  - Dots scale with zoom level
  - Optional: include ML-predicted segments (45k unscored roads) as dashed layer

Usage:
    python generate_kepler_map.py
    python generate_kepler_map.py --run outputs/run_20260627_140333
    python generate_kepler_map.py --include-ml
    python generate_kepler_map.py --output my_map.html

Output:
    outputs/speed_safety_map_full.html   (~8-10MB, all segments, no dependencies)
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False


# ── Column selection from CSV ──────────────────────────────────────────────
# All columns we want to pull from speed_safety_scores.csv for the popup
CSV_COLS = [
    "segment_id", "country", "road_class", "land_use",
    "speed_limit", "ss_limit", "speed_85th", "median_speed", "pct_over_limit",
    "sss", "sss_band",
    "sub_score_limit_alignment", "sub_score_limit_credibility", "sub_score_vru_risk",
    "credibility_class", "credibility_gap",
    "cred_raw_gap_kmh", "cred_reliability", "cred_confirmation",
    "nilsson_fatal_ratio", "nilsson_fatal_ratio_low", "nilsson_fatal_ratio_high",
    "recommended_limit", "change_effort", "est_lives_saved_RELATIVE",
    "priority_index", "priority_band",
    "exposure_score", "dist_to_school_m", "dist_to_hospital_m", "pop_density_500m",
    "sinuosity", "image_url", "country_code",
    "ml_predicted_sss", "ml_predicted_band", "ml_confidence", "is_ml_predicted",
]

# Columns to pull from ml_coverage_extension.csv when --include-ml
ML_COLS = [
    "segment_id", "country_code", "road_class_norm", "land_use",
    "speed_limit", "ss_limit",
    "ml_predicted_sss", "ml_predicted_band", "ml_confidence", "is_ml_predicted",
]


def find_latest_run(outputs_dir: Path = Path("outputs")) -> Path:
    """Return the most recent run folder."""
    runs = sorted(outputs_dir.glob("run_*/speed_safety_scores.csv"))
    if not runs:
        raise FileNotFoundError(
            f"No speed_safety_scores.csv found in {outputs_dir}/run_*/"
            "\nRun 'python main.py' first to generate outputs."
        )
    return runs[-1].parent


def load_csv(run_dir: Path) -> pd.DataFrame:
    """Load speed_safety_scores.csv with all available columns."""
    csv_path = run_dir / "speed_safety_scores.csv"
    print(f"  Loading scored segments from: {csv_path.name}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  {len(df):,} scored segments loaded")
    # Keep only columns that exist
    keep = [c for c in CSV_COLS if c in df.columns]
    missing = [c for c in CSV_COLS if c not in df.columns]
    if missing:
        print(f"  Note: {len(missing)} columns not in CSV (will be derived or omitted):")
        for c in missing[:8]:
            print(f"    {c}")
        if len(missing) > 8:
            print(f"    ...and {len(missing)-8} more")
    return df[keep].copy()


def load_centroids_from_gpkg(run_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract centroid lat/lon from speed_safety_scores_all.gpkg.
    Merges into df by segment_id.
    Falls back gracefully if geopandas not available or file missing.
    """
    gpkg_path = run_dir / "speed_safety_scores_all.gpkg"
    if not gpkg_path.exists():
        print("  GeoPackage not found — lat/lon will not be available")
        print("  Map requires latitude/longitude columns in speed_safety_scores.csv")
        return df

    if not HAS_GPD:
        print("  geopandas not installed — skipping geometry extraction")
        print("  Install with: pip install geopandas")
        return df

    print(f"  Extracting centroids from {gpkg_path.name}...")
    try:
        gdf = gpd.read_file(str(gpkg_path)).to_crs(epsg=4326)
        centroids = gdf[["segment_id"]].copy()
        centroids["latitude"] = gdf.geometry.centroid.y.round(8)
        centroids["longitude"] = gdf.geometry.centroid.x.round(8)
        n_before = len(df)
        df = df.merge(centroids, on="segment_id", how="left")
        matched = df["latitude"].notna().sum()
        print(f"  Matched {matched:,} / {n_before:,} segments to geometry")
        return df
    except Exception as e:
        print(f"  Geometry extraction failed: {e}")
        return df


def load_ml(run_dir: Path) -> pd.DataFrame | None:
    """Load ML-predicted segments from ml_coverage_extension.csv."""
    ml_path = run_dir / "ml_coverage_extension.csv"
    if not ml_path.exists():
        print("  ml_coverage_extension.csv not found — skipping ML layer")
        return None

    try:
        ml = pd.read_csv(ml_path, low_memory=False)
        keep = [c for c in ML_COLS if c in ml.columns]
        ml = ml[keep].copy()
        # Normalise column names to match scored segments
        if "road_class_norm" in ml.columns and "road_class" not in ml.columns:
            ml["road_class"] = ml["road_class_norm"]
        ml["sss"] = ml.get("ml_predicted_sss", pd.NA)
        ml["sss_band"] = ml.get("ml_predicted_band", "Unknown")
        ml["is_ml_predicted"] = True
        print(f"  {len(ml):,} ML-predicted segments loaded")
        return ml
    except Exception as e:
        print(f"  ML load failed: {e}")
        return None


def enrich(df: pd.DataFrame) -> list[dict]:
    """
    Compute derived fields used by the popup JS.
    Returns list of dicts (one per segment) ready for JSON serialisation.
    Replaces NaN/inf with None for valid JSON.
    """
    df = df.copy()

    sl  = df.get("speed_limit",  pd.Series(dtype=float))
    ss  = df.get("ss_limit",     pd.Series(dtype=float))
    f85 = df.get("speed_85th",   pd.Series(dtype=float))
    med = df.get("median_speed", pd.Series(dtype=float))

    def safe_round(s, dec=1):
        return s.round(dec).where(s.notna(), other=None)

    df["gap_f85"]   = safe_round(f85 - sl)
    df["gap_med"]   = safe_round(med - sl)
    df["spread"]    = safe_round(f85 - med)
    df["sl_vs_ss"]  = safe_round(sl - ss)

    # Country display name
    cc = df.get("country_code", pd.Series(dtype=str))
    df["country_name"] = cc.map({"MH": "Maharashtra, India", "TH": "Thailand"}).fillna("—")

    # Q1 verdict (limit direction vs SS standard)
    def q1_verdict(row):
        sl_v = row.get("speed_limit") or 0
        ss_v = row.get("ss_limit") or 0
        if not sl_v or not ss_v:
            return "NO DATA"
        if sl_v > ss_v + 5:
            return "TOO HIGH"
        if sl_v < ss_v * 0.80:
            return "TOO LOW — outdated"
        if sl_v < ss_v - 5:
            return "SLIGHTLY LOW"
        return "APPROPRIATE"

    df["q1"] = df.apply(q1_verdict, axis=1)

    # Round heavy float columns
    for col in ["sss", "priority_index", "nilsson_fatal_ratio",
                "sub_score_limit_alignment", "sub_score_limit_credibility",
                "sub_score_vru_risk", "exposure_score", "ml_predicted_sss"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Convert to list of dicts, replace NaN/inf with None
    records = df.replace({float("nan"): None, float("inf"): None,
                          float("-inf"): None}).to_dict(orient="records")
    return records


def build_html(records: list[dict], include_ml: bool = False) -> str:
    """Build the complete self-contained HTML map."""
    bands = Counter(r.get("sss_band") for r in records
                    if not r.get("is_ml_predicted"))
    scored_n = sum(1 for r in records if not r.get("is_ml_predicted"))
    ml_n = sum(1 for r in records if r.get("is_ml_predicted"))

    b_crit = bands.get("Critical", 0)
    b_high = bands.get("High Risk", 0)
    b_mod  = bands.get("Moderate", 0)
    b_acc  = bands.get("Acceptable", 0)

    data_json = json.dumps(records, separators=(",", ":"), allow_nan=False)
    data_size = len(data_json) / 1024 / 1024
    print(f"  Data payload: {data_size:.1f}MB, {len(records):,} records")

    ml_band_note = ""
    if include_ml and ml_n:
        ml_band_note = f"<br>+ {ml_n:,} ML-estimated (dashed, toggle below)"

    ml_btn = (
        f'<button class="pill pill-ml off" onclick="toggleBand(\'ML\',this)">'
        f'· · ML Est.&nbsp;<b>{ml_n:,}</b></button>'
    ) if ml_n else ''

    ml_legend_row = (
        f'<div class="lr"><div class="ld" style="background:#7c3aed;opacity:.7"></div>'
        f'ML estimate (no GPS)<span class="lc">{ml_n:,}</span></div>'
    ) if ml_n else ''

    # ── HTML template ─────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADB AI for Safer Roads — Speed Safety Score</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root {{
  --bg:#0f1117; --surface:#181c27; --surface2:#1e2333; --border:#2a3044;
  --text:#e2e8f0; --muted:#64748b; --accent:#3b82f6;
  --crit:#ef4444; --high:#f97316; --mod:#eab308; --acc:#22c55e;
  --panel-w:420px;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;height:100vh;overflow:hidden;display:flex;flex-direction:column}}
/* Top bar */
#topbar{{background:var(--surface);border-bottom:1px solid var(--border);padding:0 16px;height:48px;display:flex;align-items:center;gap:12px;flex-shrink:0;z-index:100}}
#topbar h1{{font-size:13px;font-weight:700;white-space:nowrap}}
#topbar h1 span{{color:var(--accent)}}
.pills{{display:flex;gap:5px;margin-left:auto}}
.pill{{padding:4px 11px;border-radius:20px;border:none;cursor:pointer;font-size:11px;font-weight:700;color:#fff;transition:all .15s;display:flex;align-items:center;gap:4px}}
.pill.off{{opacity:.28;filter:grayscale(1)}}
.pill-c{{background:var(--crit)}}
.pill-h{{background:var(--high)}}
.pill-m{{background:#a16207}}
.pill-a{{background:#15803d}}
.pill-ml{{background:#7c3aed}}
#cnt{{font-size:11px;color:var(--muted);background:var(--surface2);padding:3px 10px;border-radius:20px;border:1px solid var(--border);white-space:nowrap}}
/* Layout */
#main{{display:flex;flex:1;overflow:hidden}}
#map-wrap{{flex:1;position:relative;transition:flex .3s ease}}
#map{{height:100%;width:100%;background:#0f1117}}
/* Panel */
#panel{{width:0;overflow:hidden;transition:width .3s ease;background:var(--surface);border-left:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}}
#panel.open{{width:var(--panel-w)}}
#pcontent{{height:100%;overflow:hidden;display:none;flex-direction:column}}
#pcontent.show{{display:flex}}
/* Panel header */
#phdr{{padding:16px 16px 0;flex-shrink:0;position:relative}}
.pclose{{position:absolute;top:12px;right:12px;background:var(--surface2);border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:3px 8px;cursor:pointer;font-size:12px}}
.pclose:hover{{color:var(--text)}}
.ptag{{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;display:inline-block;padding:2px 8px;border-radius:3px;margin-bottom:6px}}
.pscore{{font-size:28px;font-weight:800}}
.psub{{font-size:13px;color:var(--muted)}}
.pmeta{{font-size:11px;color:var(--muted);margin:4px 0 10px}}
.pbar-bg{{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-bottom:14px}}
.pbar-fill{{height:100%;border-radius:2px;transition:width .4s ease}}
/* Divider */
.div{{height:1px;background:var(--border);margin:0 16px}}
/* Tabs */
.tabs{{display:flex;gap:2px;padding:0 16px;flex-shrink:0}}
.tab{{padding:8px 14px;font-size:12px;font-weight:600;cursor:pointer;border:none;background:none;color:var(--muted);border-bottom:2px solid transparent;transition:all .15s}}
.tab.active{{color:var(--accent);border-bottom-color:var(--accent)}}
/* Panel body */
#pbody{{flex:1;overflow-y:auto;padding:14px 16px}}
#pbody::-webkit-scrollbar{{width:3px}}
#pbody::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
.tp{{display:none}}
.tp.active{{display:block}}
/* Verdict */
.vblock{{border-radius:8px;padding:12px 14px;margin-bottom:12px;border:1px solid}}
.vok{{background:#052e16;border-color:#166534}}
.vwarn{{background:#2d1a00;border-color:#92400e}}
.vbad{{background:#2d0707;border-color:#7f1d1d}}
.vlabel{{font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;margin-bottom:4px}}
.lok{{color:#4ade80}}.lwarn{{color:#fb923c}}.lbad{{color:#f87171}}
.vdetail{{font-size:12px;color:#cbd5e1;line-height:1.5}}
.chips{{display:flex;gap:5px;margin-top:8px;flex-wrap:wrap}}
.chip{{padding:3px 8px;border-radius:4px;font-size:11px;font-weight:600;background:var(--surface2);border:1px solid var(--border)}}
.chip-over{{background:#2d0707;border-color:#ef4444;color:#f87171}}
.chip-ref{{color:var(--accent);border-color:var(--accent);background:#0c1a3a}}
/* Section */
.sec{{margin-bottom:16px}}
.seclbl{{font-size:10px;font-weight:700;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;display:flex;align-items:center;gap:6px}}
.seclbl::after{{content:'';flex:1;height:1px;background:var(--border)}}
/* Reasons */
.reason{{display:flex;gap:10px;align-items:flex-start;padding:8px 10px;border-radius:6px;margin-bottom:6px;background:var(--surface2);border-left:3px solid}}
.rc{{border-left-color:var(--crit)}}
.rw{{border-left-color:var(--high)}}
.ricon{{font-size:13px;flex-shrink:0;margin-top:1px}}
.rtext{{font-size:12px;color:#cbd5e1;line-height:1.5}}
/* Actions */
.action{{border-radius:8px;padding:11px 13px;margin-bottom:8px;background:var(--surface2);border:1px solid var(--border);border-left:3px solid}}
.au{{border-left-color:var(--crit)}}
.ap{{border-left-color:var(--high)}}
.ar{{border-left-color:var(--accent)}}
.ahdr{{display:flex;align-items:center;gap:8px;margin-bottom:4px}}
.atitle{{font-size:12px;font-weight:700;color:var(--text);flex:1}}
.abadge{{font-size:9px;font-weight:700;padding:2px 6px;border-radius:3px;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}}
.bu{{background:#2d0707;color:var(--crit)}}.bp{{background:#2d1a00;color:var(--high)}}.br{{background:#0c1a3a;color:var(--accent)}}
.adetail{{font-size:11px;color:var(--muted);line-height:1.5}}
.aevid{{font-size:10px;color:#4ade80;margin-top:3px}}
.auth{{font-size:10px;color:var(--muted);margin-top:8px;padding-top:8px;border-top:1px solid var(--border)}}
/* Score grid */
.sgrid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}}
.scell{{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:8px 10px}}
.slbl{{font-size:10px;color:var(--muted);margin-bottom:2px}}
.sval{{font-size:15px;font-weight:700}}
.sred{{color:var(--crit)}}.sblue{{color:var(--accent)}}.sgrn{{color:var(--acc)}}
/* Sub-score bars */
.sbar{{margin-bottom:10px}}
.sbhdr{{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:4px}}
.sbhdr b{{color:var(--text)}}
.sbg{{height:6px;background:var(--border);border-radius:3px;overflow:hidden}}
.sbf{{height:100%;border-radius:3px}}
.formula{{background:#0a1628;border:1px solid #1e3a5f;border-radius:6px;padding:8px 12px;font-size:11px;color:#93c5fd;font-family:'Courier New',monospace;margin-top:10px;line-height:1.6}}
.fresult{{color:#4ade80;font-weight:700;font-size:13px}}
/* Legend */
#legend{{position:absolute;bottom:16px;left:16px;z-index:500;background:rgba(24,28,39,.92);backdrop-filter:blur(8px);border:1px solid var(--border);border-radius:8px;padding:12px 14px;font-size:11px}}
.lr{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.ld{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.lc{{margin-left:auto;color:var(--muted);padding-left:14px}}
.lfoot{{margin-top:8px;padding-top:8px;border-top:1px solid var(--border);color:var(--muted);font-size:10px}}
/* Empty panel */
#pempty{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--muted);text-align:center;padding:32px;gap:8px}}
.eicon{{font-size:32px;opacity:.4}}
/* Image link */
.imglink{{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:var(--accent);text-decoration:none;padding:6px 10px;border-radius:5px;border:1px solid var(--border);margin-top:8px;background:var(--surface2)}}
.imglink:hover{{border-color:var(--accent)}}
/* ML badge */
.mlbadge{{display:inline-block;background:#3b0764;color:#c4b5fd;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;letter-spacing:.04em;margin-bottom:6px}}
</style>
</head>
<body>
<div id="topbar">
  <h1>🛡 ADB <span>Speed Safety Score</span> — Thailand &amp; Maharashtra</h1>
  <div class="pills">
    <span style="font-size:11px;color:var(--muted)">Show:</span>
    <button class="pill pill-c" onclick="toggleBand('Critical',this)">● Critical&nbsp;<b>{b_crit:,}</b></button>
    <button class="pill pill-h" onclick="toggleBand('High Risk',this)">● High Risk&nbsp;<b>{b_high:,}</b></button>
    <button class="pill pill-m" onclick="toggleBand('Moderate',this)">● Moderate&nbsp;<b>{b_mod:,}</b></button>
    <button class="pill pill-a" onclick="toggleBand('Acceptable',this)">● Acceptable&nbsp;<b>{b_acc:,}</b></button>
    {ml_btn}
  </div>
  <div id="cnt">Showing <span id="visn">{scored_n:,}</span> of {scored_n:,}{ml_band_note}</div>
</div>

<div id="main">
  <div id="map-wrap">
    <div id="map"></div>
    <div id="legend">
      <div class="lr"><div class="ld" style="background:#ef4444"></div>Critical (SSS ≥ 65)<span class="lc">{b_crit:,}</span></div>
      <div class="lr"><div class="ld" style="background:#f97316"></div>High Risk (52–65)<span class="lc">{b_high:,}</span></div>
      <div class="lr"><div class="ld" style="background:#eab308"></div>Moderate (35–52)<span class="lc">{b_mod:,}</span></div>
      <div class="lr"><div class="ld" style="background:#22c55e"></div>Acceptable (&lt;35)<span class="lc">{b_acc:,}</span></div>
      {ml_legend_row}
      <div class="lfoot">SSS = 0.20×Alignment + 0.45×Credibility + 0.35×VRU<br>{scored_n:,} GPS-confirmed segments · click any dot for details</div>
    </div>
  </div>

  <div id="panel">
    <div id="pempty">
      <div class="eicon">🗺</div>
      <div style="font-weight:600;color:var(--text)">Select a road segment</div>
      <div style="font-size:12px">Click any coloured dot to see the full Speed Safety assessment</div>
    </div>
    <div id="pcontent">
      <div id="phdr">
        <button class="pclose" onclick="closePanel()">✕ Close</button>
      </div>
      <div class="tabs">
        <button class="tab active" onclick="showTab(0)">❶❷❸ Assessment</button>
        <button class="tab" onclick="showTab(1)">📊 Score detail</button>
      </div>
      <div class="div"></div>
      <div id="pbody">
        <div id="t0" class="tp active"></div>
        <div id="t1" class="tp"></div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};

const BAND = {{
  Critical:    {{color:'#ef4444',bg:'#2d0707',text:'#f87171'}},
  'High Risk': {{color:'#f97316',bg:'#2d1400',text:'#fb923c'}},
  Moderate:    {{color:'#eab308',bg:'#1c1400',text:'#fbbf24'}},
  Acceptable:  {{color:'#22c55e',bg:'#052e16',text:'#4ade80'}},
  Unknown:     {{color:'#7c3aed',bg:'#2e1065',text:'#c4b5fd'}},
}};
const AUTH_MH = {{motorway:'NHAI',trunk:'NHAI/MSRDC',primary:'Maha. PWD',secondary:'Maha. PWD/District',tertiary:'District/Municipal'}};
const AUTH_TH = {{motorway:'DOH',trunk:'DOH',primary:'DOH',secondary:'DRR',tertiary:'DRR/Local'}};

const map = L.map('map',{{center:[14,101],zoom:5,preferCanvas:true}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'© OpenStreetMap © CARTO',subdomains:'abcd',maxZoom:19
}}).addTo(map);

const layers={{}}, visible={{}};
['Critical','High Risk','Moderate','Acceptable','ML'].forEach(b=>{{
  visible[b] = b !== 'ML';
  layers[b] = L.layerGroup();
  if (visible[b]) layers[b].addTo(map);
}});

let sel = null;

DATA.forEach(d => {{
  if (!d.latitude||!d.longitude) return;
  const isML = !!d.is_ml_predicted;
  const b = isML ? 'ML' : (d.sss_band||'Acceptable');
  const bc = BAND[b]||BAND.Acceptable;
  const baseR = b==='Critical'?8:b==='High Risk'?6:b==='Moderate'?5:isML?3:4;
  const m = L.circleMarker([d.latitude,d.longitude],{{
    radius:baseR,
    color:bc.color,fillColor:bc.color,
    fillOpacity:b==='Critical'?0.95:b==='High Risk'?0.88:isML?0.5:0.72,
    weight:isML?0.5:1.5,opacity:0.9,
    dashArray:isML?'4 3':null,
  }});
  m._d=d; m._b=b; m._r=baseR;
  m.on('click',()=>select(d,m));
  layers[b].addLayer(m);
}});

function fmt(v,u='',dec=1){{
  if(v===null||v===undefined||v!==v)return '—';
  return (+v).toFixed(dec)+u;
}}

function select(d,marker){{
  // Reset previous
  if(sel){{
    const pb=BAND[sel._b]||BAND.Acceptable;
    sel.setStyle({{color:pb.color,fillColor:pb.color,weight:sel._d.is_ml_predicted?0.5:1.5,radius:sel._r}});
  }}
  sel=marker;
  const bc=BAND[marker._b]||BAND.Acceptable;
  marker.setStyle({{color:'#fff',fillColor:bc.color,weight:3,radius:marker._r+3}});
  marker.bringToFront();

  const panel=document.getElementById('panel');
  const wasOpen=panel.classList.contains('open');
  panel.classList.add('open');
  document.getElementById('pempty').style.display='none';
  const pc=document.getElementById('pcontent');
  pc.classList.add('show');
  setTimeout(()=>map.panTo([d.latitude,d.longitude],{{animate:true,duration:0.4}}),wasOpen?0:310);

  renderPanel(d);
}}

function closePanel(){{
  document.getElementById('panel').classList.remove('open');
  const pc=document.getElementById('pcontent');
  pc.classList.remove('show');
  document.getElementById('pempty').style.display='flex';
  if(sel){{
    const pb=BAND[sel._b]||BAND.Acceptable;
    sel.setStyle({{color:pb.color,fillColor:pb.color,weight:sel._d.is_ml_predicted?0.5:1.5,radius:sel._r}});
    sel=null;
  }}
}}

function renderPanel(d){{
  const bc=BAND[d.is_ml_predicted?'Unknown':(d.sss_band||'Acceptable')];
  const sl=d.speed_limit,ss=d.ss_limit,f85=d.speed_85th,med=d.median_speed;
  const nil=d.nilsson_fatal_ratio,cc=d.country_code,rc=d.road_class_norm||d.road_class;
  const lu=d.land_use,a=d.sub_score_limit_alignment,cr=d.sub_score_limit_credibility,v=d.sub_score_vru_risk;

  // Compute live
  const gf85=f85&&sl?+(f85-sl).toFixed(1):null;
  const gmed=med&&sl?+(med-sl).toFixed(1):null;
  const spr=f85&&med?+(f85-med).toFixed(1):null;
  const slss=sl&&ss?+(sl-ss).toFixed(1):null;
  const medOver=med&&sl&&med>sl, f85Over=f85&&sl&&f85>sl+10;

  // Header
  const isML=!!d.is_ml_predicted;
  document.getElementById('phdr').innerHTML=`
    ${{isML?'<span class="mlbadge">🤖 ML ESTIMATE — No GPS data</span><br>':''}}
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
      <span class="ptag" style="background:${{bc.bg}};color:${{bc.text}}">${{d.sss_band||'Unknown'}}</span>
      <span style="font-size:11px;color:var(--muted)">${{d.country_name||d.country_code}} · ${{lu||''}} ${{rc||''}} · ${{d.segment_id||''}}</span>
    </div>
    <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">
      <span class="pscore" style="color:${{bc.color}}">${{fmt(d.sss)}}</span>
      <span class="psub">/ 100 Speed Safety Score</span>
    </div>
    <div class="pbar-bg"><div class="pbar-fill" style="width:${{Math.min(100,d.sss||0)}}%;background:${{bc.color}}"></div></div>
    <button class="pclose" onclick="closePanel()">✕</button>
  `;

  // ── TAB 0: ASSESSMENT ──────────────────────────────────────────────
  // Q1
  const q1=d.q1||'APPROPRIATE';
  let q1cls='vok',q1lcls='lok',q1icon='✓',q1lbl='APPROPRIATE';
  let q1det=`Posted ${{fmt(sl,'km/h',0)}} aligns with Safe System standard (${{fmt(ss,'km/h',0)}})`;
  if(q1==='TOO HIGH'){{
    q1cls='vbad';q1lcls='lbad';q1icon='✗';q1lbl='TOO HIGH — limit must come down';
    q1det=`Posted ${{fmt(sl,'km/h',0)}} exceeds Safe System ceiling of ${{fmt(ss,'km/h',0)}} for ${{lu}} ${{rc}}.`;
  }}else if(q1.startsWith('TOO LOW')){{
    q1cls='vwarn';q1lcls='lwarn';q1icon='⚠';q1lbl='TOO LOW — likely outdated';
    q1det=`Posted ${{fmt(sl,'km/h',0)}} is ${{fmt(Math.abs(slss||0),'km/h',0)}} below Safe System standard (${{fmt(ss,'km/h',0)}}). Road may have been upgraded — audit before raising limit.`;
  }}else if(d.credibility_class==='Non-Credible'){{
    q1cls='vwarn';q1lcls='lwarn';q1icon='⚠';q1lbl='LIMIT SET CORRECTLY — BUT IGNORED';
    q1det=`Posted ${{fmt(sl,'km/h',0)}} matches Safe System standard, but F85 is ${{fmt(gf85,'km/h',0)}} above — enforcement needed, not a limit change.`;
  }}

  const spreadNote=spr&&spr>20?`<div style="font-size:10px;color:#f97316;margin-top:5px">Spread ${{fmt(spr,'km/h',0)}} — wide speed variation signals mixed traffic; GPS probe is car-biased.</div>`:(spr?`<div style="font-size:10px;color:var(--muted);margin-top:5px">Speed spread: ${{fmt(spr,'km/h',0)}}</div>`:'');

  const q1Html=`<div class="vblock ${{q1cls}}">
    <div class="vlabel ${{q1lcls}}">${{q1icon}} ${{q1lbl}}</div>
    <div class="vdetail">${{q1det}}</div>
    <div class="chips">
      ${{sl?`<span class="chip">Posted ${{fmt(sl,'km/h',0)}}</span>`:''}}
      ${{ss?`<span class="chip chip-ref">SS Standard ${{fmt(ss,'km/h',0)}}</span>`:''}}
      ${{med?`<span class="chip ${{medOver?'chip-over':''}}">Median ${{fmt(med,'km/h',0)}}${{medOver?' ▲':''}}</span>`:''}}
      ${{f85?`<span class="chip ${{f85Over?'chip-over':''}}">F85 ${{fmt(f85,'km/h',0)}}${{f85Over?' ▲':''}}</span>`:''}}
    </div>
    ${{spreadNote}}
  </div>`;

  // Q2 Reasons
  const reas=[];
  if(gf85!==null&&gf85>15&&medOver) reas.push(['rc','🚗',`Both median (${{fmt(med,'km/h',0)}}) and F85 (${{fmt(f85,'km/h',0)}}) exceed posted ${{fmt(sl,'km/h',0)}} — systemic non-compliance across all vehicle types.`]);
  else if(gf85!==null&&gf85>15) reas.push(['rc','🚗',`F85 ${{fmt(f85,'km/h',0)}} is ${{fmt(gf85,'km/h',0)}} above posted ${{fmt(sl,'km/h',0)}} — significant proportion of drivers exceed the limit.`]);
  if(d.credibility_class==='Non-Credible') reas.push(['rc','⚠️','Limit non-credible — F85 more than 20 km/h above posted limit. Signage is effectively being ignored.']);
  if(d.credibility_class==='Infrastructure-Forced') reas.push(['rw','🔍',`Very low speeds (F85 ${{fmt(f85,'km/h',0)}}, tight spread) suggest physical calming (speed bumps). Verify on site before recommending any limit change.`]);
  if(nil&&nil>4) reas.push(['rc','❤️',`Fatal crash risk ${{fmt(nil,'×')}} the Safe System baseline (Nilsson Power Model, exponent 3.5–5.0 for Asian mixed traffic).`]);
  if(spr&&spr>20) reas.push(['rw','⇔',`Speed spread ${{fmt(spr,'km/h',0)}} — large gap between typical and fast drivers. GPS probe is car-biased; actual PTW/truck risk may be higher.`]);
  if(cc==='TH'&&['secondary','primary','tertiary'].includes(rc)) reas.push(['rw','🏍','Thailand PTW corridor — motorcycles account for 74% of road fatalities (WHO 2023). PTW risk is underweighted by car-biased GPS.']);
  if(cc==='MH'&&['primary','trunk'].includes(rc)) reas.push(['rw','🏍','Maharashtra PTW–truck conflict zone — 37% of fatalities are motorcyclists. Undivided carriageway assessed at iRAP 1–2★.']);
  if(slss&&slss>5) reas.push(['rc','↑',`Posted ${{fmt(sl,'km/h',0)}} exceeds Safe System ceiling of ${{fmt(ss,'km/h',0)}} for ${{lu}} ${{rc}}.`]);
  if(slss&&slss<-(ss||0)*0.2) reas.push(['rw','⟳',`Posted ${{fmt(sl,'km/h',0)}} is ${{fmt(Math.abs(slss),'km/h',0)}} below design speed — limit may never have been revised after road upgrade.`]);
  if(!reas.length) reas.push(['rw','✓','No major issues identified from available GPS data. Limit broadly appropriate for this road class and context.']);
  if(isML) reas.push(['rw','🤖',`This is a model estimate (XGBoost R²=0.75 on primary road features). No GPS speed data is available for this segment — use for network triage only, not individual decisions.`]);

  // Q3 Actions
  const auth=(cc==='MH'?AUTH_MH:AUTH_TH)[rc]||'Road Authority';
  const acts=[];
  if(d.change_effort==='No limit change — enforce'||d.credibility_class==='Non-Credible')
    acts.push(['au','📷','Deploy speed enforcement cameras or physical calming',`Limit is correctly set at ${{fmt(sl,'km/h',0)}} but F85 ${{fmt(f85,'km/h',0)}} shows drivers routinely ignore it. Enforcement is required, not a limit change.`,'Speed cameras reduce F85 by 5–15 km/h on treated corridors (WHO Speed Management Manual)']);
  if(slss&&slss>5)
    acts.push(['ap','🚦',`Reduce posted limit: ${{fmt(sl,'km/h',0)}} → ${{fmt(d.recommended_limit||ss,'km/h',0)}} (${{fmt(Math.abs(slss),'km/h',0)}} reduction)`,`Change effort: ${{d.change_effort||'—'}}. Aligns with Safe System standard for ${{lu}} ${{rc}}.`,'Speed limit reductions cut fatal crashes 20–40% (Nilsson Power Model, WHO)']);
  if(slss&&slss<-(ss||0)*0.2)
    acts.push(['ap','📋','Commission road audit before raising limit',`Posted ${{fmt(sl,'km/h',0)}} is well below Safe System standard (${{fmt(ss,'km/h',0)}}). Audit road geometry and infrastructure before raising — substandard roads need upgrading, not faster limits.`,'Prevents raising limits on roads that cannot safely support higher speeds']);
  if(d.credibility_class==='Infrastructure-Forced')
    acts.push(['ar','🔍','Verify physical calming features on site',`GPS evidence (F85 ${{fmt(f85,'km/h',0)}}, tight speed spread) suggests speed bumps or road tables. Confirm before recommending any limit or infrastructure change.`,'Prevents incorrect limit-change recommendation']);
  if(nil&&nil>4&&['trunk','primary'].includes(rc))
    acts.push(['ar','🛡','Install median barrier or physical separation',`Fatal crash risk ${{fmt(nil,'×')}} baseline on undivided carriageway. Head-on collision risk is the primary fatality driver.`,'Median barriers reduce head-on fatalities ~50% (iRAP Road Investment Tool)']);
  if(cc==='TH'&&['secondary','primary','tertiary'].includes(rc))
    acts.push(['ar','🏍','PTW speed enforcement + rider safety campaign',`Corridor accounts for high share of Thailand's 74% PTW fatality rate. Dedicated PTW enforcement measurably reduces risk.`,'PTW enforcement reduces PTW fatalities 20–35% (SWOV Netherlands)']);
  if(cc==='MH'&&['primary','trunk'].includes(rc))
    acts.push(['ar','🏍','Rumble strips + hard shoulder demarcation',`Undivided carriageway with PTW–truck conflict. Shoulder rumble strips prevent lane departure crashes.`,'PTW lane-departure interventions reduce fatalities 20–35%']);
  if(!acts.length)
    acts.push(['ar','📊','Schedule next audit in 12 months','No high-priority interventions identified from current GPS data. Routine monitoring maintains Safe System compliance.','Periodic re-assessment detects limit drift as traffic patterns change']);

  const imgHtml=d.image_url&&d.image_url.startsWith('http')?`<a href="${{d.image_url}}" target="_blank" class="imglink">📷 View street imagery</a>`:'';

  document.getElementById('t0').innerHTML=
    q1Html+
    `<div class="sec"><div class="seclbl">❷ Why this road is flagged</div>
    ${{reas.map(([c,ic,tx])=>`<div class="reason ${{c}}"><span class="ricon">${{ic}}</span><span class="rtext">${{tx}}</span></div>`).join('')}}
    </div>`+
    `<div class="sec"><div class="seclbl">❸ Recommended interventions</div>
    ${{acts.map(([t,ic,ti,de,ev])=>`<div class="action a${{t[1]}}">
      <div class="ahdr"><span style="font-size:15px">${{ic}}</span><span class="atitle">${{ti}}</span><span class="abadge b${{t[1]}}">${{t}}</span></div>
      <div class="adetail">${{de}}</div>
      <div class="aevid">↑ ${{ev}}</div>
    </div>`).join('')}}
    <div class="auth">Responsible authority: <b>${{auth}}</b></div>
    </div>`+
    imgHtml;

  // ── TAB 1: SCORE DETAIL ──────────────────────────────────────────────
  const pctOver=d.pct_over_limit!==null&&d.pct_over_limit!==undefined?fmt(d.pct_over_limit,'%',1):'—';
  const distSch=d.dist_to_school_m!==null?fmt(d.dist_to_school_m,'m',0):'—';
  const distHosp=d.dist_to_hospital_m!==null?fmt(d.dist_to_hospital_m,'m',0):'—';
  const expScore=d.exposure_score!==null?fmt(d.exposure_score,'',1):'—';
  const nilLow=d.nilsson_fatal_ratio_low,nilHigh=d.nilsson_fatal_ratio_high;
  const nilRange=nilLow&&nilHigh?` <span style="color:var(--muted);font-size:12px">[${{fmt(nilLow,'',1)}}–${{fmt(nilHigh,'',1)}}×]</span>`:'';

  document.getElementById('t1').innerHTML=`
    <div class="sec"><div class="seclbl">SSS component scores</div>
      <div class="sbar">
        <div class="sbhdr"><span>Alignment <span style="color:var(--muted)">(posted vs Safe System) · weight 20%</span></span><b>${{fmt(a)}}/100</b></div>
        <div class="sbg"><div class="sbf" style="width:${{Math.min(100,a||0)}}%;background:#3b82f6"></div></div>
      </div>
      <div class="sbar">
        <div class="sbhdr"><span>Credibility gap <span style="color:var(--muted)">(dual F85+median) · weight 45%</span></span><b>${{fmt(cr)}}/100</b></div>
        <div class="sbg"><div class="sbf" style="width:${{Math.min(100,cr||0)}}%;background:#ef4444"></div></div>
      </div>
      <div class="sbar">
        <div class="sbhdr"><span>VRU context risk <span style="color:var(--muted)">(PTW-weighted) · weight 35%</span></span><b>${{fmt(v)}}/100</b></div>
        <div class="sbg"><div class="sbf" style="width:${{Math.min(100,v||0)}}%;background:#f97316"></div></div>
      </div>
      <div class="formula">
        SSS = 0.20×${{fmt(a)}} + 0.45×${{fmt(cr)}} + 0.35×${{fmt(v)}} = <span class="fresult">${{fmt(d.sss)}}</span> → ${{d.sss_band}}
      </div>
    </div>

    <div class="sec"><div class="seclbl">GPS behaviour evidence</div>
      <div class="sgrid">
        <div class="scell"><div class="slbl">Posted limit</div><div class="sval sblue">${{fmt(sl,'km/h',0)}}</div></div>
        <div class="scell"><div class="slbl">Safe System standard</div><div class="sval">${{fmt(ss,'km/h',0)}}</div></div>
        <div class="scell"><div class="slbl">Median GPS speed</div><div class="sval ${{medOver?'sred':''}}">${{fmt(med,'km/h',0)}}</div></div>
        <div class="scell"><div class="slbl">F85 GPS speed</div><div class="sval ${{f85Over?'sred':''}}">${{fmt(f85,'km/h',0)}}</div></div>
        <div class="scell"><div class="slbl">Speed spread (F85−median)</div><div class="sval ${{spr&&spr>20?'sred':''}}">${{fmt(spr,'km/h',0)}}</div></div>
        <div class="scell"><div class="slbl">% vehicles over limit</div><div class="sval">${{pctOver}}</div></div>
        <div class="scell"><div class="slbl">Credibility class</div><div class="sval" style="font-size:11px">${{d.credibility_class||'—'}}</div></div>
        <div class="scell"><div class="slbl">Nilsson fatal risk</div><div class="sval ${{nil&&nil>4?'sred':''}}">${{fmt(nil,'×')}}${{nilRange}}</div></div>
      </div>
    </div>

    <div class="sec"><div class="seclbl">Exposure context</div>
      <div class="sgrid">
        <div class="scell"><div class="slbl">Priority Index (E×L×S)</div><div class="sval">${{fmt(d.priority_index,'',1)}}</div></div>
        <div class="scell"><div class="slbl">Priority band</div><div class="sval" style="font-size:12px;color:${{BAND[d.priority_band||'Acceptable']?.color||'#888'}}">${{d.priority_band||'—'}}</div></div>
        <div class="scell"><div class="slbl">Exposure score</div><div class="sval">${{expScore}}</div></div>
        <div class="scell"><div class="slbl">Population density</div><div class="sval" style="font-size:12px">${{d.pop_density_500m?fmt(d.pop_density_500m,'ppl/km²',0):'—'}}</div></div>
        <div class="scell"><div class="slbl">Nearest school</div><div class="sval" style="font-size:12px">${{distSch}}</div></div>
        <div class="scell"><div class="slbl">Nearest hospital</div><div class="sval" style="font-size:12px">${{distHosp}}</div></div>
      </div>
      <div style="font-size:10px;color:var(--muted);margin-top:4px">Priority Index answers "where to act first" — complementary to SSS, not the same question.</div>
    </div>

    <div style="padding:8px 10px;background:var(--surface2);border-radius:6px;font-size:10px;color:var(--muted);line-height:1.6">
      <b style="color:#64748b">Sources:</b> ADB GPS probe · WHO Speed Management Manual 2023 · iRAP · Nilsson 2004 · Elvik 2009 · WorldPop · HOTOSM
    </div>`;
}}

function showTab(n){{
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i===n));
  document.querySelectorAll('.tp').forEach((p,i)=>p.classList.toggle('active',i===n));
}}

function toggleBand(band,btn){{
  visible[band]=!visible[band];
  btn.classList.toggle('off',!visible[band]);
  if(visible[band]) map.addLayer(layers[band]);
  else map.removeLayer(layers[band]);
  const tot=['Critical','High Risk','Moderate','Acceptable','ML']
    .filter(b=>visible[b]).reduce((s,b)=>s+layers[b].getLayers().length,0);
  document.getElementById('visn').textContent=tot.toLocaleString();
}}

map.on('zoom',()=>{{
  const z=map.getZoom();
  const sc=z<7?1:z<9?1.4:z<11?2:2.8;
  const base={{Critical:8,'High Risk':6,Moderate:5,Acceptable:4,ML:3}};
  Object.keys(layers).forEach(b=>{{
    layers[b].getLayers().forEach(m=>m.setRadius((base[b]||4)*sc));
  }});
}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate full Speed Safety Score map with all 14,711 segments"
    )
    parser.add_argument(
        "--run", default=None,
        help="Path to run folder (e.g. outputs/run_20260627_140333). Auto-detects latest if omitted."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output HTML path. Default: <run_dir>/speed_safety_map_full.html"
    )
    parser.add_argument(
        "--include-ml", action="store_true",
        help="Also show ML-predicted segments (45k unscored roads) as a dashed layer"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  SPEED SAFETY MAP — Full Coverage Generator")
    print("="*60)

    # Find run directory
    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = find_latest_run()
    print(f"\nRun directory: {run_dir}")

    # Load scored segments
    print("\n[1/4] Loading speed_safety_scores.csv...")
    df = load_csv(run_dir)

    # Add lat/lon from GeoPackage
    print("\n[2/4] Extracting centroids from GeoPackage...")
    df = load_centroids_from_gpkg(run_dir, df)

    # Check we have coordinates
    n_with_coords = df["latitude"].notna().sum() if "latitude" in df.columns else 0
    if n_with_coords == 0:
        print("\nERROR: No lat/lon coordinates found.")
        print("Ensure speed_safety_scores_all.gpkg is in the run directory,")
        print("or that latitude/longitude columns exist in speed_safety_scores.csv.")
        sys.exit(1)
    print(f"  {n_with_coords:,} segments have coordinates")

    # Normalise road_class column (CSV uses 'road_class', popup uses 'road_class_norm')
    if "road_class" in df.columns and "road_class_norm" not in df.columns:
        df["road_class_norm"] = df["road_class"]

    # Optionally add ML segments
    ml_df = None
    if args.include_ml:
        print("\n[3/4] Loading ML-predicted segments...")
        ml_df = load_ml(run_dir)
        if ml_df is not None:
            # Add centroids to ML segments too
            ml_df = load_centroids_from_gpkg(run_dir, ml_df)
    else:
        print("\n[3/4] ML layer skipped (use --include-ml to add 45k unscored segments)")

    # Combine and enrich
    print("\n[4/4] Building HTML map...")
    all_dfs = [df]
    if ml_df is not None and len(ml_df):
        all_dfs.append(ml_df)

    combined = pd.concat(all_dfs, ignore_index=True) if len(all_dfs) > 1 else df
    records = enrich(combined)

    html = build_html(records, include_ml=(ml_df is not None and len(ml_df) > 0))

    # Write output
    output_path = Path(args.output) if args.output else run_dir / "speed_safety_map_full.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"  ✅ Map saved: {output_path}")
    print(f"  Size: {size_mb:.1f}MB  |  Segments: {len(records):,}")
    print(f"\n  Open in any browser — no server needed.")
    print(f"  For GitHub: push to simsim branch and link via GitHub Pages.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()