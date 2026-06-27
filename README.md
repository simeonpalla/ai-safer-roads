# Speed Safety Score — ADB AI for Safer Roads 2026

Quantifies whether posted speed limits are appropriate relative to the WHO Safe System standard, confirmed by GPS probe behaviour, and prioritised by population exposure. Covers **Maharashtra** (India) and **Thailand** from the ADB dataset.

---

## Quick Start

```bash
pip install -r requirements.txt
python download_pois.py          # one-time OSM POI fetch (~2 min)
python main.py                   # full pipeline (~45 min without Mapillary)
```

Outputs land in `outputs/run_<timestamp>/`. Key files: `speed_safety_map.html` (interactive map) and `Top_Priority_Interventions.xlsx` (policy brief).

> **First time?** See [Data Setup](#data-setup) — several raster files must be downloaded manually.

---

## What the pipeline produces

| Output | Description |
|--------|-------------|
| `speed_safety_map.html` | Interactive map — 1,000 highest-risk roads with full popups (❶ Is the limit right? ❷ Why? ❸ Intervention) |
| `speed_safety_scores_all.gpkg` | **All 14,711 scored segments** with every column — load in QGIS/ArcGIS for full network view |
| `speed_safety_scores.csv` | Flat CSV of all Tier 2 scored segments |
| `Top_Priority_Interventions.xlsx` | Policy brief — Critical and High Risk segments with intervention narrative |
| `ml_coverage_extension.gpkg/csv` | XGBoost-predicted SSS for 45,183 unscored segments (R²=0.750 generalisation) |
| `score_overview.png` | Score distribution, band breakdown, sensitivity analysis chart |
| `scatter_sss_vs_pct_over_limit.png` | Hidden Danger quadrant: 4,028 roads (27.4%) high-risk but compliant |
| `ai_anomaly_segments.csv` | Isolation Forest flagged segments — experimental, analyst triage only |

**Note on map coverage:** The HTML map shows 500 segments per country (all Critical first, then High Risk). For full 14,711-segment coverage use `speed_safety_scores_all.gpkg` in QGIS, or the Kepler.gl link in the repo.

---

## Scoring methodology

### Speed Safety Score (SSS = 0.20×A + 0.45×C + 0.35×V)

Three sub-scores reflecting different aspects of limit appropriateness:

| Sub-score | Weight | What it measures |
|-----------|--------|-----------------|
| **Alignment (A)** | 20% | Gap between posted limit and WHO Safe System standard for this road type × land use. Two-sided: too high AND too low both score as misaligned |
| **Credibility Gap (C)** | 45% | Dual-signal: F85 excess modulated by spread-based reliability weight (0.5–1.0, mixed-traffic dampener) AND median confirmation factor (0.4–1.0). Not F85 alone |
| **VRU Context Risk (V)** | 35% | Pedestrian/PTW exposure: urban/rural multiplier + PTW country weight (Thailand 74% PTW fatalities; Maharashtra primary/trunk 37%) + VIIRS nighttime boost |

**Safe System limits** follow WHO Speed Management Manual + iRAP star ratings. 8 of 17 road class × land use cells are VERIFIED direct citations; 9 are INTERPOLATED (explicitly flagged, alignment sub-score dampened 30%). Rural primary/secondary set at 60 km/h for undivided roads (iRAP 1–2★, MH/TH context).

**Geometry adjustment:** Sinuosity ≥1.20 triggers −10 to −25 km/h downward adjustment per AASHTO Green Book. Applied to 5,803 segments (8.3%).

**Score bands:** Critical ≥65 | High Risk 52–65 | Moderate 35–52 | Acceptable <35

**Weight robustness:** Mean Spearman ρ = 0.9675 under ±10% weight perturbation (>0.95 threshold).

### Priority Index (secondary — "where to act first")

Exposure × Likelihood × Severity geometric mean. Answers a different question from SSS: not "is this limit appropriate" but "given limited budgets, which road first." Spearman ρ = 0.51 vs SSS — correlated but adds independent information (26% top-20% overlap).

Components: WorldPop population density · HOTOSM schools/hospitals · OSM markets, transit, religious, universities, crossings · Nilsson Power Model · speed variability · road class severity · VIIRS nighttime exposure.

### AI / ML layers

| Layer | Status | Detail |
|-------|--------|--------|
| **XGBoost coverage extension** | Production | Predicts SSS for 45,183 unscored segments. R²=0.993 (GPS features, leakage diagnostic) / R²=0.750 (primary features only — honest generalisation estimate). Map toggle layer |
| **Isolation Forest anomaly detection** | Experimental | Finds statistically unusual roads vs same-class peers. 2,207 flagged (15%). Not used in primary scoring or map — analyst CSV only |
| **Road geometry (sinuosity)** | Production | Actual/crow-flies path ratio from LineString. Adjusts Safe System limit per AASHTO |
| **VIIRS nighttime lights** | Production | NASA annual composite radiance → nighttime VRU exposure proxy |
| **Mapillary street-level** | Optional | Queries Mapillary v4 API per segment bbox. High-SSS + zero coverage = infrastructure blindspot flag. Pass `--mapillary-token` |

---

## Key findings (run_20260627_140333)

- **14,711 segments** fully scored (Tier 2 GPS confirmed); **14,761** alignment-only (Tier 1)
- **471 Critical** (3.2%) · **2,831 High Risk** (19.2%) · **6,248 Moderate** (42.5%) · **5,161 Acceptable** (35.1%)
- **7,552 segments** need speed limit reduction; average reduction: **25.8 km/h**
- **8,420 segments** at >2× Nilsson fatal-crash baseline; **5,542** at >4×
- **4,028 Hidden Danger segments** (27.4%) — high SSS but low % over limit; missed by volume-based monitoring
- Conventional speed-camera monitoring misses **77% of high-risk roads**
- **1,339 segments** need enforcement (not limit change) — limit correctly set but drivers ignore it
- Est. **178.2 lives/year** saved if all limits corrected (range 89–356) — illustrative, NOT validated
- **165 intervention zones** covering 3,157 segments
- XGBoost R²=**0.750** (generalisation, primary features only; R²=0.993 with GPS features = leakage diagnostic)
- Weight stability: mean Spearman ρ=**0.9675** under ±10% perturbation (all 6 tests pass)
- SSS uniquely flags **2,453 high-risk segments** that traffic-volume ranking would miss (83% non-overlap)

---

## Project structure

```
main.py                  Pipeline entry point
config.py                Weights (0.20/0.45/0.35), thresholds, band cuts, Safe System limits
scoring.py               SSS computation — three sub-scores with dual-signal credibility
advanced_scoring.py      Nilsson, credibility classes, recommended limits, lives saved, intervention zones
preprocessing.py         Data loading, normalisation, segment ID assignment
enrichment.py            WorldPop, HOTOSM, 7 POI types, OSM infra tags, exposure layer
geometry_features.py     Sinuosity + bearing stddev from road geometry
ghsl_features.py         GHSL SMOD settlement classification (7-level urbanicity)
viirs_features.py        VIIRS nighttime lights enrichment
priority_scoring.py      Exposure × Likelihood × Severity priority index
ml_extension.py          XGBoost coverage extension + Isolation Forest anomaly detection
mapillary_features.py    Mapillary v4 API street-level infrastructure enrichment
evaluation.py            Sensitivity analysis, weight stability, cross-country consistency
visualization.py         Folium interactive map + scatter chart + GeoPackage/CSV export
policy_brief.py          Top_Priority_Interventions.xlsx generation
logger.py                Shared logging setup
download_pois.py         One-time OSM POI download
extract_osm_data.py      Extract schools/hospitals/road infra from OSM PBF files
data/                    GeoJSON inputs + helmet Excel  [gitignored — ADB provided]
enrichment_data/         Rasters and POI files  [gitignored — see Data Setup]
outputs/                 One timestamped folder per run  [gitignored]
```

---

## Data Setup

Large files are gitignored. Run once before first pipeline run.

### 1. ADB data (required)
```
data/ADB_Innovation_Maharashtra.geojson
data/ADB_Innovation_Thailand.geojson
data/Archive/Road_Safety_Performance_Indicators_(Helmet_Wearing_results)_(adb_dashboard_data_v02).xlsx
```

### 2. WorldPop population rasters (required)
Download 1 km resolution from [worldpop.org](https://www.worldpop.org/geodata/listing?id=75):
```
enrichment_data/worldpop_MH_2020_1km.tif
enrichment_data/worldpop_TH_2020_1km.tif
```

### 3. GHSL SMOD settlement classification (optional — improves ML)
Download from [EU Copernicus HSL](https://human-settlement.emergency.copernicus.eu/):
```
enrichment_data/ghsl/GHS_SMOD_E2025.tif
```

### 4. VIIRS nighttime lights (optional)
Download VNL v2 annual composite from [eogdata.mines.edu](https://eogdata.mines.edu/products/vnl/):
```
enrichment_data/viirs/viirs_ntl.tif
```

### 5. OSM schools and hospitals (required)
From [Humanitarian OSM Team](https://data.humdata.org/organization/hot):
```
enrichment_data/schools/schools_MH.geojson   enrichment_data/schools/schools_TH.geojson
enrichment_data/hospitals/hospitals_MH.geojson   enrichment_data/hospitals/hospitals_TH.geojson
```

### 6. OSM POI data — auto-download
```bash
python download_pois.py   # markets, transit, religious, university, crossings (~2 min)
```

### 7. Mapillary API token (optional)
Free token at [mapillary.com/developer](https://www.mapillary.com/developer). Pass at runtime: `--mapillary-token "MLY|..."`

---

## Reproduction

```bash
# Install
pip install -r requirements.txt

# Full pipeline (matches published results)
python main.py

# With Mapillary blindspot detection (~5-6 hr extra)
python main.py --mapillary-token "MLY|YOUR_TOKEN_HERE"

# Skip map + eval (faster debugging)
python main.py --no-map --no-eval

# Skip ML extension
python main.py --no-ml
```

**Runtime:** ~45 min (without Mapillary). Python 3.9+ required.

---

## Viewing full results

The HTML map shows the top 1,000 segments. For full 14,711-segment coverage:

1. **QGIS (recommended):** Open `speed_safety_scores_all.gpkg` → style by `sss_band` column
2. **Kepler.gl:** Drag `speed_safety_scores.csv` to [kepler.gl](https://kepler.gl) → color by `sss_band`
3. **GitHub link for submission:** `https://github.com/simeonpalla/ai-safer-roads/tree/simsim`

---

## Limitations

- GPS probe data is **car-biased** — truck and PTW speeds underrepresented. Spread-based reliability weight partially mitigates this
- **79% of network unscored** — no GPS data or posted limit. Unscored ≠ safe
- Lives saved figures are **illustrative** — VKM conversion constant unverified
- Nilsson exponent uncertainty: 3.5–5.0 for Asian mixed traffic (Elvik 2009)
- Safe System limits for 9 road class × land use combinations are **interpolated** estimates, not direct WHO citations