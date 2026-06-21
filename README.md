# Speed Safety Score — ADB AI for Safer Roads 2026

Quantifies whether posted speed limits are appropriate relative to the Safe System standard, prioritised by population exposure. Covers **Maharashtra** (India) and **Thailand** datasets provided by ADB.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Outputs land in `outputs/run_<timestamp>/`. The key file is `speed_safety_map.html` — open it in any browser.

### Optional: satellite and street-level enrichment

```bash
# With Mapillary street-level infrastructure features
python main.py --mapillary-token "MLY|YOUR_TOKEN_HERE"

# Skip VIIRS nighttime-lights enrichment (default: attempts local file)
python main.py --no-viirs
```

For VIIRS, download the annual composite from `eogdata.mines.edu/products/vnl/` and place at `enrichment_data/viirs/viirs_ntl.tif`.

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
| `ml_validation_scatter.png` | XGBoost OOF validation scatter (R²=0.817, RMSE=7.95) |
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

Components: WorldPop population density · HOTOSM schools and hospitals · Nilsson Power Model fatal-crash risk · speed variability · road class severity.

### AI / ML layers

1. **XGBoost coverage extension** — predicts SSS for 45,183 Tier 1 segments lacking behavioural speed data. 5-fold CV RMSE=7.95, R²=0.817. Shown as a separate toggle layer on the map.
2. **Isolation Forest anomaly detection** — finds statistically unusual roads relative to similar-class peers. Status: experimental; off-map, for analyst triage only.
3. **Road geometry (sinuosity)** — computes actual/crow-flies path ratio from LineString coordinates. Adjusts Safe System limit per AASHTO Green Book. Status: production.
4. **Mapillary street-level infrastructure** — queries Mapillary v4 API per road segment bbox. Returns traffic sign density and road object counts. High-SSS segments with zero coverage are flagged as *infrastructure blindspots*. Status: production-ready (pass `--mapillary-token`).
   - API discovery: 0.01°×0.01° grid cells (~1 km²) complete in 3–15s. Larger 0.09° cells time out in dense Bangkok (>100k features/cell). Rural Maharashtra returned 0 features across 220 sampled cells — an infrastructure blindspot finding in itself.
   - Bangkok coverage confirmed: 1,400–1,700 features per km² cell (Sukhumvit, Silom corridors).
5. **VIIRS nighttime lights** — samples NASA/NOAA VIIRS annual composite radiance as a proxy for informal market activity and nighttime pedestrian exposure. Status: code complete; requires local GeoTIFF.

---

## Key findings (current run)

- **14,711 segments** fully scored (Tier 2); **14,761** alignment-only (Tier 1)
- **833 Critical** (5.7%) · **5,175 High Risk** (35.2%) · **4,219 Moderate** (28.7%) · **4,484 Acceptable** (30.5%)
- **7,297 segments** need speed limit reduction; average reduction required: **22.2 km/h**
- **7,561 segments** at >2× Nilsson baseline fatal-crash risk; **4,317** at >4×
- **946 high-risk segments** (SSS≥40) in the bottom 25% by traffic volume — flagged by this model, missed by volume-based prioritisation
- **5,803 curved segments** had Safe System limits reduced for geometry (sinuosity)
- Rank stability: mean Spearman ρ=0.99 under ±10% weight perturbations

---

## Project structure

```
main.py                  Pipeline entry point — run this
config.py                All weights, thresholds, band cuts
scoring.py               SSS computation (three sub-scores)
advanced_scoring.py      Nilsson, credibility, lives-saved, intervention zones
enrichment.py            WorldPop, schools, hospitals exposure layer
priority_index.py        Exposure × Likelihood × Severity
ml_coverage.py           XGBoost extension + Isolation Forest
mapillary_features.py    Mapillary API enrichment module
viirs_features.py        VIIRS nighttime lights enrichment
map_builder.py           Folium interactive map
export_policy_brief.py   Top_Priority_Interventions.xlsx
data/                    GeoJSON inputs (MH, TH) + helmet Excel
enrichment_data/         WorldPop GeoTIFFs, HOTOSM school/hospital points
outputs/                 One timestamped folder per pipeline run
```

---

## Reproduction

```bash
# Full pipeline with all optional enrichments
python main.py --mapillary-token "MLY|YOUR_TOKEN"

# Skip map generation (faster run, all other outputs still produced)
python main.py --no-map

# Skip evaluation/sensitivity analysis
python main.py --no-eval

# Skip both VIIRS and ML coverage extension (fastest run)
python main.py --no-viirs --no-ml
```

Requirements: Python 3.9+, see `requirements.txt`.
