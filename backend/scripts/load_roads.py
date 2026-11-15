"""Build the walkable street network from a local OSM extract and load it into PostGIS
as road_nodes / road_edges.

    docker compose run --rm tools bash scripts/fetch_osm.sh        # once: makes data/raw/usc_roads.osm.pbf
    docker compose run --rm tools python -m scripts.load_roads [path.osm.pbf]

Uses a local file (not the public Overpass API, which is often overloaded).
Includes sidewalks/footpaths and the OSM `lit` tag (a direct streetlight signal).
"""
import argparse
import glob
import os
import subprocess
import sys
import time
import warnings

import geopandas as gpd
import h3
import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon
from sqlalchemy import text

from app import create_app
from app.extensions import db
from app.services.cell_features import refresh_static_features
from app.services.h3_utils import circle_ring, line_cells

# Midpoints/sampling in lng/lat are accurate enough for picking H3 cells on short edges
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

EXTRA_TAGS = ["lit", "sidewalk", "footway", "foot", "access", "service"]
# Mirrors OSMnx's "walk" network filter
NOT_WALKABLE_HIGHWAY = {"abandoned", "bus_guideway", "construction", "cycleway", "motor", "motorway",
                        "motorway_link", "no", "planned", "platform", "proposed", "raceway", "razed"}
# ...except that an explicit foot tag wins. USC's main walks (Trousdale Pkwy, Childs Way,
# Watt Way, Hellman Way, Bloom Walk, W 34th St, ...) are highway=cycleway + foot=designated,
# and some campus paths are access=private + foot=yes. Dropping them left the planner no
# way across campus, so routes zig-zagged around it on the few plain footways.
FOOT_ALLOWED = {"yes", "designated", "permissive"}


def flat(v):
    """OSMnx merges attributes of simplified edges into lists."""
    if isinstance(v, list):
        return ";".join(str(x) for x in dict.fromkeys(v))
    if v is None or (isinstance(v, float) and v != v):
        return None
    return str(v)


def walkable(d):
    foot = flat(d.get("foot"))
    if foot == "no" or d.get("area") == "yes":
        return False
    if foot in FOOT_ALLOWED:   # explicitly open to pedestrians: cycleway, private service road, ...
        return d.get("highway") not in NOT_WALKABLE_HIGHWAY - {"cycleway"}
    return not (d.get("highway") in NOT_WALKABLE_HIGHWAY
                or d.get("access") == "private" or d.get("service") == "private")


def find_input(data_dir):
    hits = sorted(glob.glob(os.path.join(data_dir, "raw", "*.osm.pbf")))
    if not hits:
        sys.exit(f"no .osm.pbf in {data_dir}/raw: run `bash scripts/fetch_osm.sh` first")
    return hits[0]


def build_graph(pbf, circle, work):
    xml = os.path.join(work, "roads.osm")
    subprocess.run(["osmium", "cat", pbf, "-f", "osm", "-o", xml, "--overwrite"], check=True)
    G = ox.graph_from_xml(xml, bidirectional=True, simplify=False, retain_all=True)
    os.remove(xml)
    drop = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if not walkable(d)]
    G.remove_edges_from(drop)
    G = ox.truncate.truncate_graph_polygon(G, circle, truncate_by_edge=True)
    G = ox.truncate.largest_component(G)
    G = ox.simplify_graph(G)
    # graph_from_xml(simplify=False) skips this; needed for intersection counts
    nx.set_node_attributes(G, ox.stats.count_streets_per_node(G), name="street_count")
    return ox.convert.to_undirected(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="OSM .pbf (default: first *.osm.pbf in data/raw)")
    args = ap.parse_args()

    app = create_app()
    with app.app_context():
        cfg = app.config
        res = cfg["H3_RES"]
        ox.settings.useful_tags_way = list(dict.fromkeys(ox.settings.useful_tags_way + EXTRA_TAGS))
        work = os.path.join(cfg["DATA_DIR"], "work")
        os.makedirs(work, exist_ok=True)

        pbf = args.path or find_input(cfg["DATA_DIR"])
        # study circle + margin, so routes that start near the edge still have streets
        ring = circle_ring(cfg["USC_LAT"], cfg["USC_LNG"], cfg["RADIUS_KM"] + cfg["ROAD_MARGIN_KM"])
        circle = Polygon([(lng, lat) for lat, lng in ring])
        t = time.time()
        print(f"building walk network from {pbf} ...")
        G = build_graph(pbf, circle, work)
        nodes, edges = ox.graph_to_gdfs(G)
        print(f"  {len(nodes):,} nodes, {len(edges):,} edges in {time.time() - t:.0f}s")

        nodes = nodes.reset_index()
        node_out = gpd.GeoDataFrame({
            "osmid": nodes["osmid"].astype("int64"),
            "street_count": nodes.get("street_count"),
            "h3_cell": [h3.latlng_to_cell(y, x, res) for y, x in zip(nodes["y"], nodes["x"])],
        }, geometry=nodes.geometry.values, crs=4326).rename_geometry("geom")

        edges = edges.reset_index()
        mids = edges.geometry.interpolate(0.5, normalized=True)
        edge_out = gpd.GeoDataFrame({
            "u": edges["u"].astype("int64"),
            "v": edges["v"].astype("int64"),
            "key": edges["key"].astype("int16"),
            **{col: edges[col].map(flat) if col in edges else None
               for col in ("osmid", "name", "highway", "lit", "sidewalk", "footway")},
            "length_m": edges["length"].astype(float).round(2),
            "h3_cell": [h3.latlng_to_cell(p.y, p.x, res) for p in mids],
            # Postgres array literal: geopandas bulk-loads via COPY, which can't take Python lists
            "h3_cells": ["{" + ",".join(line_cells(g, L)) + "}" for g, L in zip(edges.geometry, edges["length"])],
        }, geometry=edges.geometry.values, crs=4326).rename_geometry("geom")

        db.session.execute(text("TRUNCATE road_edges, road_nodes RESTART IDENTITY"))
        db.session.commit()
        node_out.to_postgis("road_nodes", db.engine, if_exists="append", index=False, chunksize=20000)
        edge_out.to_postgis("road_edges", db.engine, if_exists="append", index=False, chunksize=20000)
        refresh_static_features()
        lit_known = edge_out["lit"].notna().mean() * 100
        print(f"  loaded {len(node_out):,} nodes / {len(edge_out):,} edges "
              f"({lit_known:.1f}% of edges carry an OSM lit tag)")


if __name__ == "__main__":
    main()
