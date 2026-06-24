"""
download_pois.py — Fetch VRU-generating POIs from OpenStreetMap (Overpass API).

Run once before the pipeline:
    python download_pois.py

Saves GeoJSON files to enrichment_data/<type>/<type>_TH.geojson etc.
enrichment.py loads them automatically on the next pipeline run.

POI types fetched:
  markets    — markets, bazaars, supermarkets (heavy pedestrian footfall)
  transit    — bus stations, railway stations (road-crossing hotspots)
  religious  — places of worship (temples/wats in TH, temples/mosques in MH)
  university — universities and colleges (dense motorcycle/bicycle commuters)
  crossings  — railway level crossings (extremely high fatality risk in India)
"""

import json
import time
import sys
from pathlib import Path

import geopandas as gpd
import requests

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

DATA_FILES = {
    "TH": "data/ADB_Innovation_Thailand.geojson",
    "MH": "data/ADB_Innovation_Maharashtra.geojson",
}

# OSM queries — nodes only for dense types (no ways) to keep response size down.
# For transit: bus_station + railway=station only (not every highway=bus_stop
# which would return millions of nodes and time out for large areas).
POI_QUERIES = {
    "markets": """(
  node["amenity"="marketplace"];
  node["amenity"="market"];
  node["shop"="supermarket"];
  node["shop"="mall"];
  way["amenity"="marketplace"]; way["amenity"="market"];
); out center;""",

    "transit": """(
  node["amenity"="bus_station"];
  node["railway"="station"];
  node["railway"="halt"];
  way["amenity"="bus_station"]; way["railway"="station"];
); out center;""",

    "religious": """(
  node["amenity"="place_of_worship"];
  way["amenity"="place_of_worship"];
); out center;""",

    "university": """(
  node["amenity"="university"];
  node["amenity"="college"];
  way["amenity"="university"]; way["amenity"="college"];
); out center;""",

    "crossings": """(
  node["railway"="level_crossing"];
); out body;""",

    # OSM schools — saved into enrichment_data/schools/ alongside HOTOSM files.
    # _load_amenities() merges all geojsons in the folder automatically.
    "schools_osm": """(
  node["amenity"="school"];
  node["amenity"="kindergarten"];
  node["amenity"="childcare"];
  way["amenity"="school"];
  way["amenity"="kindergarten"];
); out center;""",

    # OSM hospitals/clinics — saved into enrichment_data/hospitals/
    "hospitals_osm": """(
  node["amenity"="hospital"];
  node["amenity"="clinic"];
  node["amenity"="health_post"];
  node["healthcare"="hospital"];
  node["healthcare"="clinic"];
  way["amenity"="hospital"];
  way["amenity"="clinic"];
); out center;""",
}

# Types that should land in a different output folder than their own name.
# e.g. schools_osm -> enrichment_data/schools/schools_osm_MH.geojson
# so _load_amenities("enrichment_data/schools/") picks up both HOTOSM + OSM.
OUTPUT_FOLDER_OVERRIDE = {
    "schools_osm":   "schools",
    "hospitals_osm": "hospitals",
}

# Buffer added to road data bbox before querying (degrees)
BBOX_BUFFER = 0.05


def _get_road_bbox(country_code: str) -> dict:
    """Compute tight bbox from actual road segment geometries."""
    path = Path(DATA_FILES[country_code])
    if not path.exists():
        print(f"  Road data not found: {path}")
        return None
    print(f"  Reading road extent from {path.name}...")
    gdf = gpd.read_file(path)
    b = gdf.total_bounds   # (minx, miny, maxx, maxy) = (west, south, east, north)
    return {
        "south": round(b[1] - BBOX_BUFFER, 3),
        "west":  round(b[0] - BBOX_BUFFER, 3),
        "north": round(b[3] + BBOX_BUFFER, 3),
        "east":  round(b[2] + BBOX_BUFFER, 3),
    }


def _overpass_query(query_body: str, bbox: dict, timeout: int = 90) -> list:
    """Run Overpass QL query within bbox. Returns list of elements."""
    s, w, n, e = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    area_deg2 = (n - s) * (e - w)
    print(f"    bbox: {s},{w} -> {n},{e}  ({area_deg2:.2f} sq deg)")

    full_query = f"[out:json][timeout:{timeout}][bbox:{s},{w},{n},{e}];\n{query_body}"

    for url in OVERPASS_ENDPOINTS:
        try:
            resp = requests.get(url, params={"data": full_query}, timeout=timeout + 20)
            if resp.status_code == 200:
                elements = resp.json().get("elements", [])
                print(f"    {len(elements):,} elements from {url.split('/')[2]}")
                return elements
            print(f"    {url.split('/')[2]} returned HTTP {resp.status_code} -- trying next")
        except requests.Timeout:
            print(f"    {url.split('/')[2]} timed out -- trying next")
        except Exception as exc:
            print(f"    {url.split('/')[2]} error: {exc} -- trying next")

    print("    All endpoints failed")
    return []


def _elements_to_geojson(elements: list) -> dict:
    features = []
    for el in elements:
        if el["type"] == "node":
            lon, lat = el.get("lon"), el.get("lat")
        elif "center" in el:
            lon, lat = el["center"].get("lon"), el["center"].get("lat")
        else:
            continue
        if lon is None or lat is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": el.get("tags", {}),
        })
    return {"type": "FeatureCollection", "features": features}


def main():
    out_root = Path("enrichment_data")

    for poi_type, query_body in POI_QUERIES.items():
        folder = OUTPUT_FOLDER_OVERRIDE.get(poi_type, poi_type)
        (out_root / folder).mkdir(parents=True, exist_ok=True)

        for country_code in ["TH", "MH"]:
            out_file = out_root / folder / f"{poi_type}_{country_code}.geojson"

            if out_file.exists():
                existing = gpd.read_file(out_file)
                print(f"[{poi_type}/{country_code}] Already exists ({len(existing):,} features) -- skipping")
                continue

            print(f"\n[{poi_type}/{country_code}] Fetching...")
            bbox = _get_road_bbox(country_code)
            if bbox is None:
                continue

            elements = _overpass_query(query_body, bbox)
            if not elements:
                print(f"[{poi_type}/{country_code}] No results -- skipping")
                continue

            fc = _elements_to_geojson(elements)
            n = len(fc["features"])
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(fc, f)
            print(f"[{poi_type}/{country_code}] Saved {n:,} features -> {out_file}")

            time.sleep(3)   # polite delay between Overpass requests

    print("\nDone. Re-run the pipeline to include new proximity features.")


if __name__ == "__main__":
    # Ensure stdout handles unicode on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
