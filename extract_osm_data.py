"""
extract_osm_data.py — Extract schools, hospitals, AND road infrastructure
attributes from OSM PBF files.

Run this ONCE to generate the GeoJSON files needed by enrichment.py.
Uses the OSM PBF files you already downloaded from Geofabrik.

Usage:
    python extract_osm_data.py

Reads from:
    openSourceData/thailand-260612.osm.pbf
    openSourceData/india-260612.osm.pbf   (if available)
    OR: any *india* or *maharashtra* .osm.pbf in openSourceData/

Writes to:
    enrichment_data/schools/schools_TH.geojson
    enrichment_data/schools/schools_MH.geojson
    enrichment_data/hospitals/hospitals_TH.geojson
    enrichment_data/hospitals/hospitals_MH.geojson
    enrichment_data/road_infra/road_infra_TH.geojson   (NEW)
    enrichment_data/road_infra/road_infra_MH.geojson   (NEW)

ROAD INFRA EXTRACTION (NEW — Priority Index reviewer feedback):
  The Severity layer's road-class component previously relied entirely on
  a static country/road-class assumption (ROAD_CLASS_SEVERITY_MAP in
  config.py — "trunk roads are probably undivided, motorways are probably
  divided"). This extractor pulls the ACTUAL OSM tags that the assumption
  was standing in for, so enrichment.py can use observed road geometry
  facts instead, falling back to the assumption only where no OSM match is
  found:
    lanes      — numeric lane count (more lanes = wider crossing distance,
                 generally higher design speed)
    oneway     — "yes" means no opposing-direction traffic on this carriageway
                 (eliminates head-on collision risk — a major severity factor)
    surface    — paved vs unpaved (unpaved correlates with loss-of-control crashes)
    lit        — street lighting present (absence worsens night crash severity,
                 a standard iRAP risk factor)
    junction   — "roundabout" replaces higher-energy angle/head-on crash types
                 at junctions with lower-energy glancing collisions
  These are real, observed facts from the map data, not assumptions about
  what a "trunk road" probably looks like in a given country.
"""

import json
import osmium
import sys
from pathlib import Path

# ── Maharashtra bounding box (lon_min, lat_min, lon_max, lat_max) ──────────
MH_BBOX = (72.5, 15.5, 80.9, 22.1)
TH_BBOX = (97.3,  5.5, 105.7, 20.5)

SCHOOL_TAGS   = {"amenity": ["school", "college", "university", "kindergarten"]}
HOSPITAL_TAGS = {"amenity": ["hospital", "clinic", "doctors", "health_post"],
                 "healthcare": ["hospital", "clinic", "doctor", "health_centre"]}

# Road classes worth extracting infrastructure tags for — matches the
# road_class_norm vocabulary already used elsewhere in the pipeline
# (config._normalize_road_class / ROAD_CLASS_SEVERITY_MAP), plus link variants.
ROAD_HIGHWAY_TAGS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "unclassified", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}


class AmenityExtractor(osmium.SimpleHandler):
    """Extract point locations of schools and hospitals from OSM PBF."""

    def __init__(self, bbox, school_out, hospital_out):
        super().__init__()
        self.bbox        = bbox   # (minx, miny, maxx, maxy)
        self.schools     = []
        self.hospitals   = []
        self.school_out  = school_out
        self.hospital_out= hospital_out

    def _in_bbox(self, lon, lat):
        minx, miny, maxx, maxy = self.bbox
        return minx <= lon <= maxx and miny <= lat <= maxy

    def _check_tags(self, tags):
        """Return 'school', 'hospital', or None based on OSM tags."""
        amenity = tags.get("amenity", "")
        healthcare = tags.get("healthcare", "")

        if amenity in SCHOOL_TAGS["amenity"]:
            return "school"
        if amenity in HOSPITAL_TAGS["amenity"] or healthcare in HOSPITAL_TAGS["healthcare"]:
            return "hospital"
        return None

    def node(self, n):
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        if not self._in_bbox(lon, lat):
            return

        tags = {t.k: t.v for t in n.tags}
        kind = self._check_tags(tags)
        if kind is None:
            return

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "osm_id":   str(n.id),
                "name":     tags.get("name", tags.get("name:en", "")),
                "amenity":  tags.get("amenity", ""),
                "healthcare": tags.get("healthcare", ""),
                "type":     kind,
            },
        }
        if kind == "school":
            self.schools.append(feature)
        else:
            self.hospitals.append(feature)

    def way(self, w):
        """For ways (building polygons), use the centroid approximation."""
        try:
            # Get first node as representative point
            tags = {t.k: t.v for t in w.tags}
            kind = self._check_tags(tags)
            if kind is None:
                return
            # We can't get coordinates from ways without location cache
            # This is handled by node() above — most amenities are nodes
        except Exception:
            pass

    def save(self):
        for path, features, label in [
            (self.school_out,   self.schools,   "schools"),
            (self.hospital_out, self.hospitals, "hospitals"),
        ]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            geojson = {
                "type": "FeatureCollection",
                "features": features,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(geojson, f)
            print(f"  Saved {len(features):,} {label} -> {path}")


class RoadInfraExtractor(osmium.SimpleHandler):
    """
    Extract road way geometries with infrastructure tags (lanes, oneway,
    surface, lit, junction) from OSM PBF.

    Requires locations=True (node coordinates resolved onto each way) —
    slower and more memory-intensive than the point-amenity pass above
    (AmenityExtractor uses locations=False since it only needs node
    coordinates directly), so this runs as its own separate pass over the
    PBF file.
    """

    def __init__(self, bbox, out_path):
        super().__init__()
        self.bbox     = bbox
        self.out_path = out_path
        self.ways     = []

    def _in_bbox(self, lon, lat):
        minx, miny, maxx, maxy = self.bbox
        return minx <= lon <= maxx and miny <= lat <= maxy

    def way(self, w):
        tags = {t.k: t.v for t in w.tags}
        highway = tags.get("highway", "")
        if highway not in ROAD_HIGHWAY_TAGS:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except Exception:
            return
        if len(coords) < 2:
            return
        # Cheap bbox filter on the first node — good enough to drop ways
        # entirely outside the country bbox without checking every node.
        if not self._in_bbox(*coords[0]):
            return

        lanes_raw = tags.get("lanes", "")
        try:
            lanes = float(lanes_raw.split(";")[0].split(",")[0])
        except (ValueError, AttributeError):
            lanes = None

        self.ways.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "osm_id":   str(w.id),
                "highway":  highway,
                "lanes":    lanes,
                "oneway":   tags.get("oneway", ""),
                "surface":  tags.get("surface", ""),
                "lit":      tags.get("lit", ""),
                "junction": tags.get("junction", ""),
            },
        })

    def save(self):
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        geojson = {"type": "FeatureCollection", "features": self.ways}
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)
        print(f"  Saved {len(self.ways):,} road ways -> {self.out_path}")


def find_pbf(folder, keywords):
    """Find a PBF file matching any of the keywords."""
    folder = Path(folder)
    for keyword in keywords:
        matches = list(folder.glob(f"*{keyword}*.osm.pbf"))
        if matches:
            return matches[0]
    return None


def extract_country(pbf_path, bbox, country_code, out_dir="enrichment_data"):
    print(f"\n[{country_code}] Extracting from: {pbf_path.name}")
    print(f"  Bounding box: {bbox}")

    school_out     = f"{out_dir}/schools/schools_{country_code}.geojson"
    hospital_out   = f"{out_dir}/hospitals/hospitals_{country_code}.geojson"
    road_infra_out = f"{out_dir}/road_infra/road_infra_{country_code}.geojson"

    # Schools + hospitals (point amenities)
    if Path(school_out).exists() and Path(hospital_out).exists():
        ns = len(json.load(open(school_out))["features"])
        nh = len(json.load(open(hospital_out))["features"])
        print(f"  Already extracted: {ns} schools, {nh} hospitals - skipping")
    else:
        handler = AmenityExtractor(bbox, school_out, hospital_out)
        print(f"  Scanning PBF file for schools/hospitals (1-3 minutes for country files)...")
        handler.apply_file(str(pbf_path), locations=False)
        handler.save()
        ns, nh = len(handler.schools), len(handler.hospitals)

    # Road infrastructure (lanes/oneway/surface/lit/junction) — NEW.
    # Needs locations=True to resolve way geometry, so it's slower than the
    # amenity pass above; expect this to take longer on large country files.
    if Path(road_infra_out).exists():
        nr = len(json.load(open(road_infra_out))["features"])
        print(f"  Already extracted: {nr} road ways - skipping")
    else:
        road_handler = RoadInfraExtractor(bbox, road_infra_out)
        print(f"  Scanning PBF file for road infrastructure tags "
              f"(slower - resolves node locations onto ways)...")
        road_handler.apply_file(str(pbf_path), locations=True)
        road_handler.save()
        nr = len(road_handler.ways)

    return ns, nh, nr


def main():
    osm_dir = Path("openSourceData")
    out_dir  = "enrichment_data"

    print("=" * 60)
    print("  OSM DATA EXTRACTOR - Schools, Hospitals & Road Infra")
    print("=" * 60)

    if not osm_dir.exists():
        print(f"ERROR: '{osm_dir}' folder not found.")
        print("Make sure you run this from your adb/ project folder.")
        sys.exit(1)

    results = {}

    # Thailand
    th_pbf = find_pbf(osm_dir, ["thailand", "tha"])
    if th_pbf:
        ns, nh, nr = extract_country(th_pbf, TH_BBOX, "TH", out_dir)
        results["TH"] = (ns, nh, nr)
    else:
        print("\n[TH] No Thailand PBF found in openSourceData/")
        print("  Download from: https://download.geofabrik.de/asia/thailand-latest.osm.pbf")

    # India / Maharashtra
    mh_pbf = find_pbf(osm_dir, ["maharashtra", "india", "ind", "western-zone", "western"])
    if mh_pbf:
        ns, nh, nr = extract_country(mh_pbf, MH_BBOX, "MH", out_dir)
        results["MH"] = (ns, nh, nr)
    else:
        print("\n[MH] No India/Maharashtra PBF found in openSourceData/")
        print("  Download Maharashtra only (120MB) from:")
        print("  https://download.geofabrik.de/asia/india/maharashtra-latest.osm.pbf")

    print("\n" + "=" * 60)
    print("  EXTRACTION COMPLETE")
    for cc, (ns, nh, nr) in results.items():
        print(f"  {cc}: {ns:,} schools, {nh:,} hospitals, {nr:,} road ways extracted")
    print("\n  GeoJSON files saved to enrichment_data/schools/, enrichment_data/hospitals/,")
    print("  and enrichment_data/road_infra/")
    print("  Run main.py to use them in enrichment scoring.")
    print("=" * 60)


if __name__ == "__main__":
    main()
