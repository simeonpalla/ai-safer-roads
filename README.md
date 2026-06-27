# Speed Safety Score — ADB AI for Safer Roads 2026

Quantifies whether posted speed limits are appropriate relative to the Safe System standard, prioritised by population exposure. Covers **Maharashtra** (India) and **Thailand** datasets provided by ADB.

---

## Quick Start

```bash
pip install -r requirements.txt
python download_pois.py          # one-time OSM POI fetch (~2 min)
python main.py                   # full pipeline, ~45 min without Mapillary
```

Outputs land in `outputs/run_<timestamp>/`. The key file is `speed_safety_map.html` — open it in any browser.

> **First time?** See [Data Setup](#data-setup) below — several raster files (WorldPop, GHSL, VIIRS) must be downloaded manually before running.

---

## What the pipeline produces

| Output | Description |
|--------|-------------|
| `speed_safety_map.html` | Interactive map — all scored roads, colour-coded by band, with popups |
| `speed_safety_scores_all.gpkg` | GeoPackage with every scored segment and all columns |
| `speed_safety_scores.csv` | Flat CSV of Tier 2 scored segments |
| `Top_Priority_Interventions.xlsx` | Policy brief — Top 20 priority corridors with intervention narrative |
| `ml_coverage_extension.gpkg` | XGBoost-predicted SSS for 45,183 unscored segments |
| `score_overview.png` | Diagnostic chart: score distribution, band breakdown, sensitivity |
| `ml_validation_scatter.png` | XGBoost OOF validation scatter (R²=0.877, RMSE=6.06) |
| `ai_anomaly_segments.csv` | Isolation Forest flagged segments (experimental) |

---

## Scoring methodology

### Level 1 — Speed Safety Score (SSS)

Three sub-scores, equal-weighted:

| Sub-score | What it measures |
|-----------|-----------------|
| **Safe System Alignment** (38%) | Gap between posted limit and Safe System standard for road type + land use |
| **Limit Credibility** (30%) | Whether the posted limit matches observed 85th-percentile speed |
| **VRU Risk Context** (32%) | Pedestrian/cyclist exposure based on land use, road class, helmet compliance |

Safe System limits follow WHO / iRAP protocol. **Geometry adjustment**: sinuosity ≥1.20 triggers a −10 to −25 km/h downward adjustment per AASHTO Green Book, capped at 30 km/h floor. Applied to 5,803 segments (8.3% of scored network).

### Level 2 — Priority Index

Exposure × Likelihood × Severity geometric mean — answers "where to act first," not "is this limit appropriate."

Components: WorldPop population density · HOTOSM schools and hospitals · OSM markets, transit, religious sites, universities, railway crossings · Nilsson Power Model fatal-crash risk · speed variability · road class severity · VIIRS nighttime exposure.

### AI / ML layers

1. **XGBoost coverage extension** — predicts SSS for 45,183 segments lacking behavioural speed data. 5-fold CV RMSE=6.06, R²=0.877. Features include GHSL settlement class, VIIRS NTL score, sinuosity, and `dist_to_nearest_vru_attractor_m`. Shown as a separate toggle layer on the map.
2. **Isolation Forest anomaly detection** — finds statistically unusual roads relative to similar-class peers. Status: experimental; off-map, for analyst triage only.
3. **Road geometry (sinuosity)** — computes actual/crow-flies path ratio from LineString coordinates. Adjusts Safe System limit per AASHTO Green Book. Status: production.
4. **Mapillary street-level infrastructure** — queries Mapillary v4 API per road segment bbox. Returns traffic sign density and road object counts. High-SSS segments with zero coverage are flagged as *infrastructure blindspots*. Status: production-ready (pass `--mapillary-token`).
   - API discovery: 0.01°×0.01° grid cells (~1 km²) complete in 3–15s. Larger 0.09° cells time out in dense Bangkok (>100k features/cell). Rural Maharashtra returned 0 features across 220 sampled cells — an infrastructure blindspot finding in itself.
   - Bangkok coverage confirmed: 1,400–1,700 features per km² cell (Sukhumvit, Silom corridors).
5. **VIIRS nighttime lights** — samples NASA/NOAA VIIRS annual composite radiance as a proxy for informal market activity and nighttime pedestrian exposure. Status: code complete; requires local GeoTIFF.

---

## Key findings (current run)

- **14,711 segments** fully scored (Tier 2); **14,761** alignment-only (Tier 1)
- **679 Critical** (4.6%) · **5,121 High Risk** (34.8%) · **3,790 Moderate** (25.8%) · **5,121 Acceptable** (34.8%)
- **7,459 segments** need speed limit reduction; average reduction required: **21.8 km/h**
- **8,010 segments** at >2× Nilsson baseline fatal-crash risk; **4,427** at >4×
- **5,162 Hidden Danger segments** (SSS≥45, <40% over limit) — high-risk but compliant; missed by volume-based monitoring
- Conventional monitoring misses **75%** of high-risk roads
- **12,625 segments** covered by Mapillary; **2,710 high-risk blindspots** (no street-level imagery)
- **5,800 priority segments** in policy brief (679 Critical + 5,121 High Risk)
- Est. **160.9 lives/year** saved if limits corrected (range 80–322) — illustrative, not validated
- **309 intervention zones** covering 5,980 segments
- XGBoost R²=**0.877**, RMSE=**6.06** — up from 0.817/8.0 baseline
- SHAP top driver: Safe System Speed Limit (mean |SHAP|=8.38)
- Rank stability: mean Spearman ρ=0.99 under ±10% weight perturbations

---

## Project structure

```
main.py                  Pipeline entry point — run this
config.py                All weights, thresholds, band cuts
scoring.py               SSS computation (three sub-scores)
advanced_scoring.py      Nilsson, credibility, lives-saved, intervention zones
enrichment.py            WorldPop, HOTOSM, 7 POI types, OSM infra exposure layer
ghsl_features.py         GHSL SMOD settlement classification (7-level urbanicity)
priority_scoring.py      Exposure × Likelihood × Severity priority index
ml_extension.py          XGBoost coverage extension + Isolation Forest anomaly
mapillary_features.py    Mapillary v4 API street-level infrastructure enrichment
viirs_features.py        VIIRS nighttime lights enrichment
download_pois.py         One-time OSM POI download (markets, transit, religious, university, crossings)
logger.py                Shared logging setup
map_builder.py           Folium interactive map
export_policy_brief.py   Top_Priority_Interventions.xlsx
data/                    GeoJSON inputs (MH, TH) + helmet Excel  [not in repo — ADB provided]
enrichment_data/         Rasters and POI files  [large files not in repo — see Data Setup]
outputs/                 One timestamped folder per pipeline run  [gitignored]
```

---

## Data Setup

All large data files are gitignored. Run these steps once before the first pipeline run.

### 1. ADB challenge data (required)
Place the two GeoJSONs and the helmet Excel provided by ADB into `data/`:
```
data/ADB_Innovation_Maharashtra.geojson
data/ADB_Innovation_Thailand.geojson
data/Archive/Road_Safety_Performance_Indicators_(Helmet_Wearing_results)_(adb_dashboard_data_v02).xlsx
```

### 2. WorldPop population rasters (required)
Download 1 km resolution population counts for Maharashtra and Thailand from [worldpop.org](https://www.worldpop.org/geodata/listing?id=75) and place at:
```
enrichment_data/worldpop_MH_2020_1km.tif
enrichment_data/worldpop_TH_2020_1km.tif
```

### 3. GHSL SMOD settlement classification (required for full R²=0.877)
Download `GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0.tif` from the
[EU Copernicus Human Settlement Layer](https://human-settlement.emergency.copernicus.eu/) and place at:
```
enrichment_data/ghsl/GHS_SMOD_E2025.tif
```
Without this file the pipeline falls back to the binary URBAN/RURAL field (R² ~0.82).

### 4. VIIRS nighttime lights (required for NTL features)
Download the annual composite (VNL v2, 2024) from [eogdata.mines.edu/products/vnl/](https://eogdata.mines.edu/products/vnl/) and place at:
```
enrichment_data/viirs/viirs_ntl.tif
```

### 5. OSM schools and hospitals (required)
Download from [Humanitarian OpenStreetMap Team](https://data.humdata.org/organization/hot):
```
enrichment_data/schools/schools_MH.geojson
enrichment_data/schools/schools_TH.geojson
enrichment_data/hospitals/hospitals_MH.geojson
enrichment_data/hospitals/hospitals_TH.geojson
```

### 6. OSM POI data — markets, transit, religious, university, crossings (auto-download)
Run once to fetch all five POI types for both countries from the Overpass API (~2 min):
```bash
python download_pois.py
```
This writes to `enrichment_data/{markets,transit,religious,university,crossings}/`. Already-downloaded files are skipped on re-runs.

### 7. Mapillary API token (optional — required for blindspot layer)
Get a free token at [mapillary.com/developer](https://www.mapillary.com/developer) and pass it at runtime (see below). The pipeline runs without it; Mapillary features will be zero-filled.

---

## Reproduction

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline — matches published results (R²=0.877)
python main.py --mapillary-token "MLY|YOUR_TOKEN_HERE"

# Without Mapillary (all other features intact, ~4 hr faster)
python main.py

# Faster run — skip map generation and evaluation charts
python main.py --no-map --no-eval

# Minimum run — no VIIRS, no ML extension
python main.py --no-viirs --no-ml
```

**Expected runtime** (with Mapillary, full enrichment): ~5–6 hours, dominated by Mapillary API queries for ~3,900 grid cells. Without Mapillary: ~45 minutes.

Requirements: Python 3.9+, see `requirements.txt`.
