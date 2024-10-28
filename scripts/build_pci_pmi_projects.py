# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2021-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build PCI and PMI projects from interactive PCI-PMI Transparency Platform,
based on their respective types.

https://ec.europa.eu/energy/infrastructure/transparency_platform/map-viewer/main.html
"""
import json
import logging
import re
from itertools import chain

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import tqdm
from _helpers import configure_logging, set_scenario_config
from clean_pci_pmi_projects import _split_multilinestring
from pypsa.geo import haversine_pts  # to recalculate crow-flies distance
from shapely import segmentize, unary_union
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import MultiPoint, Point
from shapely.ops import nearest_points

logging.getLogger("pyogrio._io").setLevel(logging.WARNING)  # disable pyogrio info
logger = logging.getLogger(__name__)

DISTANCE_CRS = "EPSG:3035"
GEO_CRS = "EPSG:4326"
OFFSHORE_BUS_RADIUS = 5000
CLUSTER_TOL = 25000

COLUMNS_LINES = [
    "bus0",
    "bus1",
    "length",
    "build_year",
    "s_nom",
    "v_nom",
    "num_parallel",
    "carrier",
    "type",
    "tags",
]
COLUMNS_LINKS = [
    "bus0",
    "bus1",
    "length",
    "build_year",
    "p_nom",
    "carrier",
    "underwater_fraction",
    "tags",
]
COLUMNS_STORAGE_UNITS = [
    "bus",
    "build_year",
    "p_nom",
    "max_hours",
    "carrier",
    "tags",
]
COLUMNS_STORES = [
    "bus",
    "build_year",
    "e_nom_max",
    "carrier",
    "tags",
]


def _map_endpoints_to_closest_region(
    gdf,
    regions,
    max_distance=OFFSHORE_BUS_RADIUS,
    coords=0,
    lines=True,
):
    if lines:
        gdf_points = gdf.geometry.apply(lambda x: Point(x.coords[coords]))
    else:
        gdf_points = gdf.geometry

    gdf_points = gpd.GeoDataFrame(geometry=gdf_points, crs=GEO_CRS)
    # Spatial join nearest with regions

    # Find nearest region index
    regions = regions.to_crs(DISTANCE_CRS)
    gdf_points = gdf_points.to_crs(DISTANCE_CRS)

    gdf_points = gpd.sjoin_nearest(gdf_points, regions, how="left")
    gdf_points = gdf_points.join(
        regions, on="name", lsuffix="_point", rsuffix="_region"
    )
    gdf_points["distance"] = gdf_points.apply(
        lambda x: x.geometry_point.distance(x.geometry_region), axis=1
    )

    bool_too_far = gdf_points["distance"] > max_distance
    gdf_points.loc[bool_too_far, "name"] = None

    return gdf_points["name"]


def _map_to_closest_region(
    gdf, regions, max_distance=OFFSHORE_BUS_RADIUS, add_suffix=None
):
    # add Suffix to regions index
    regions = regions.copy()
    if add_suffix:
        regions.index = regions.index + " " + add_suffix

    gdf = gdf.copy()
    # if columns bus0 and bus1 dont exist, create them
    if "bus0" not in gdf.columns:
        gdf["bus0"] = None
    if "bus1" not in gdf.columns:
        gdf["bus1"] = None

    # Apply mapping to rows where 'bus0' is None
    gdf.loc[gdf["bus0"].isna(), "bus0"] = _map_endpoints_to_closest_region(
        gdf[gdf["bus0"].isna()], regions, max_distance, coords=0
    )

    # Apply mapping to rows where 'bus1' is None
    gdf.loc[gdf["bus1"].isna(), "bus1"] = _map_endpoints_to_closest_region(
        gdf[gdf["bus1"].isna()], regions, max_distance, coords=-1
    )

    return gdf


def _map_points_to_closest_region(
    gdf,
    regions,
    max_distance=OFFSHORE_BUS_RADIUS,
    add_suffix=None,
    distance_crs=DISTANCE_CRS,
    geo_crs=GEO_CRS,
):
    # Change name of index in regions to "bus"
    regions = regions.copy()
    if add_suffix:
        regions.index = regions.index + " " + add_suffix

    if "bus" not in gdf.columns:
        gdf["bus"] = None

    gdf.loc[gdf["bus"].isna(), "bus"] = _map_endpoints_to_closest_region(
        gdf[gdf["bus"].isna()],
        regions,
        max_distance,
        coords=0,
        lines=False,
    )

    return gdf


def _simplify_lines_to_380(lines, linetype):
    lines = lines.copy()
    logger.info("Mapping all AC lines onto a single 380 kV layer")

    lines["type"] = linetype
    lines["v_nom"] = 380
    i_nom_380 = pypsa.Network().line_types["i_nom"][linetype_380]
    lines.loc[:, "num_parallel"] = lines["s_nom"] / (
        np.sqrt(3) * lines["v_nom"] * i_nom_380
    )

    return lines


def _drop_redundant_lines_links(lines):
    logger.info("Dropping lines/links that are internal or missing endpoints.")
    bool_missing_bus = lines["bus0"].isna() | lines["bus1"].isna()
    bool_internal_line = lines["bus0"] == lines["bus1"]

    lines = lines[~bool_missing_bus & ~bool_internal_line]

    return lines


def _add_geometry_to_tags(df):
    df.loc[:, "tags"] = df.apply(
        lambda row: {**row["tags"], "geometry": row["geometry"]}, axis=1
    )

    return df


def _calculate_haversine_distance(n, lines, line_length_factor):
    coords = n.buses[["x", "y"]]

    lines.loc[:, "length"] = (
        haversine_pts(coords.loc[lines["bus0"]], coords.loc[lines["bus1"]])
        * line_length_factor
    ).round(1)

    return lines


def _set_underwater_fraction(links, regions_offshore):
    links = links.copy()
    links.loc[:, "underwater_fraction"] = (
        links.intersection(regions_offshore.union_all()).to_crs(DISTANCE_CRS).length
        / links.to_crs(DISTANCE_CRS).length
    ).round(2)

    return links


def _create_new_buses(
    gdf,
    regions,
    scope,
    carrier,
    distance_crs=DISTANCE_CRS,
    geo_crs=GEO_CRS,
    tol=OFFSHORE_BUS_RADIUS,
):
    buffered_regions = (
        regions.to_crs(distance_crs)
        .buffer(5000)
        .to_crs(geo_crs)
        .union_all()  # Coastal buffer
    )

    # filter all rows in gdf where at least one of the geometry linestring endings is outside unary_union(regions)
    # create a list of Points of all linestring endings in gdf
    list_points = list(
        chain(*gdf.geometry.apply(lambda x: [Point(x.coords[0]), Point(x.coords[-1])]))
    )
    # create multipoint geometry of all points
    list_points = MultiPoint(list_points)
    list_points = list_points.intersection(scope.union_all())
    # Drop all points that are within unary_union(regions) and a buffer of 5000 meters
    list_points = list_points.difference(buffered_regions)

    gdf_points = gpd.GeoDataFrame(
        geometry=[geom for geom in list_points.geoms], crs=geo_crs
    )

    gdf_points["geometry"] = gdf_points.to_crs(distance_crs).buffer(tol).to_crs(geo_crs)

    # Aggregate rows with touching polygons
    gdf_points = gdf_points.dissolve()
    # split into separate polygons
    gdf_points = gdf_points.explode().reset_index(drop=True)

    gdf_points["poi"] = (
        gdf_points["geometry"]
        .to_crs(distance_crs)
        .apply(lambda polygon: polylabel(polygon, tolerance=tol / 2))
        .to_crs(geo_crs)
    )

    # Extract x and y coordinates into separate columns
    gdf_points["x"] = gdf_points["poi"].x
    gdf_points["y"] = gdf_points["poi"].y
    gdf_points["name"] = gdf_points.apply(
        lambda x: f"PCI-PMI {int(x.name)+1} {carrier}", axis=1
    )
    gdf_points.set_index("name", inplace=True)
    gdf_points["carrier"] = carrier
    gdf_points["location"] = gdf_points.index

    return gdf_points[["x", "y", "carrier", "location", "geometry"]]


def _find_points_on_line_overpassing_region(
    link, regions, distance_crs=DISTANCE_CRS, geo_crs=GEO_CRS
):

    overlap = gpd.overlay(link, regions)

    # All rows with multilinestrings, split them into their individual linestrings and fill the rows with the same data
    overlap = pd.concat(
        overlap.apply(_split_multilinestring, axis=1).tolist(), ignore_index=True
    )

    overlap["center_point"] = overlap["geometry"].apply(
        lambda l: l.interpolate(l.length / 2)
    )

    overlap["on_point"] = overlap.apply(
        lambda row: nearest_points(row["center_point"], row["geometry"])[1], axis=1
    )

    return overlap[["on_point"]].rename(columns={"on_point": "geometry"})


def _split_to_overpassing_segments(
    gdf, regions, distance_crs=DISTANCE_CRS, geo_crs=GEO_CRS
):
    logger.info("Splitting linestrings into segments that connect overpassing regions.")
    buffer_radius = 50  # m

    ## Delete later
    gdf_split = gdf.copy().to_crs(distance_crs)
    regions_dist = regions.to_crs(distance_crs)

    # Increase resolution of both geometries
    gdf_split["geometry"] = gdf_split["geometry"].apply(lambda x: segmentize(x, 200))
    regions_dist["geometry"] = regions_dist["geometry"].apply(
        lambda x: segmentize(x, 300)
    )

    gdf_points = _find_points_on_line_overpassing_region(gdf_split, regions_dist)
    gdf_points = gpd.GeoDataFrame(gdf_points, crs=distance_crs)

    gdf_points["buffer"] = gdf_points["geometry"].buffer(buffer_radius)

    # Split linestrings of gdf by union of points[buffer]
    gdf_split["geometry"] = gdf_split["geometry"].apply(
        lambda x: x.difference(gdf_points["buffer"].union_all())
    )

    # Drop empty geometries
    gdf_split = gdf_split[~gdf_split["geometry"].is_empty]

    gdf_split.reset_index(inplace=True)
    # All rows with multilinestrings, split them into their individual linestrings and fill the rows with the same data
    gdf_split = pd.concat(
        gdf_split.apply(_split_multilinestring, axis=1).tolist(), ignore_index=True
    )

    gdf_split = gpd.GeoDataFrame(gdf_split, geometry="geometry", crs=distance_crs)

    # Drop empty geometries
    gdf_split = gdf_split[~gdf_split["geometry"].is_empty]

    # Recalculate lengths
    gdf_split["length"] = (
        gdf_split["geometry"].length.div(1e3).round(1)
    )  # Calculate in km, round to 1 decimal

    gdf_split.to_crs(geo_crs, inplace=True)
    gdf_split.set_index("id", inplace=True)

    return gdf_split


def _set_unique_index(gdf):
    gdf = gdf.copy()
    gdf.reset_index(inplace=True)

    gdf["id"] = gdf["id"].apply(lambda x: x.split("-")[1])
    gdf["id"] = "PCI" + "-" + gdf["id"]

    # Group by id and sort from North to south, west to east
    gdf = gdf.sort_values(
        by=["id", "geometry"],
    )

    # Group by id and add cumcount +1 and zero padding
    gdf["id"] = (
        gdf["id"] + "-" + gdf.groupby("id").cumcount().add(1).astype(str).str.zfill(2)
    )

    gdf.set_index("id", inplace=True)

    return gdf


def _cluster_close_buses(
    gdf, tol=CLUSTER_TOL, distance_crs=DISTANCE_CRS, geo_crs=GEO_CRS
):
    prefix = gdf.index[0].split(" ")[0]
    suffix = gdf.index[0].split(" ")[-1]
    carrier = gdf["carrier"].iloc[0]

    logger.info(f"Clustering close PCI (offshore) buses within {tol} m.")

    gdf.to_crs(distance_crs, inplace=True)
    gdf["cluster_buffer"] = gdf["geometry"].buffer(tol)

    cluster_union = gdf["cluster_buffer"].union_all()
    gdf_clusters = gpd.GeoDataFrame(
        geometry=list(cluster_union.geoms),
        crs=distance_crs,
    )

    # spatial join
    gdf = gpd.sjoin(gdf, gdf_clusters, how="left", predicate="intersects")

    # Group by index_right
    gdf = gdf.groupby("index_right").agg(
        {
            "geometry": lambda x: unary_union(x).convex_hull,
            "carrier": "first",
        }
    )

    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=distance_crs)

    gdf["poi"] = gdf.apply(
        lambda x: polylabel(x["geometry"], tolerance=tol / 2), axis=1
    )
    gdf["poi"].crs = distance_crs

    gdf["geometry"] = gdf["geometry"].to_crs(geo_crs)
    gdf["poi"] = gdf["poi"].to_crs(geo_crs)
    gdf.to_crs(geo_crs, inplace=True)

    gdf["x"] = gdf["poi"].x
    gdf["y"] = gdf["poi"].y

    gdf["name"] = prefix + " " + (gdf.index + 1).astype(str) + " " + suffix
    gdf["carrier"] = carrier
    gdf["location"] = gdf.index
    gdf.set_index("name", inplace=True)

    return gdf[["x", "y", "carrier", "location", "geometry"]]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_pci_pmi_projects",
            ll="vopt",
            clusters=90,
            opts="",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    line_length_factor = snakemake.params.line_length_factor
    haversine_distance = snakemake.params["pci_pmi_projects"]["haversine_distance"]
    linetype_380 = snakemake.config["lines"]["types"][380]

    n = pypsa.Network(snakemake.input.network)
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    regions_offshore = gpd.read_file(snakemake.input.regions_offshore).set_index("name")
    scope = gpd.read_file(snakemake.input.scope)

    logger.info("Imported network.")

    # TODO: make this more robust
    pci_pmi_projects = dict(
        zip(
            list(snakemake.params["pci_pmi_projects"]["include"].keys()),
            snakemake.input.pci_pmi_projects,
        )
    )

    # Load data
    components = {}

    for component, path in pci_pmi_projects.items():
        components[component] = gpd.read_file(path)
        components[component].set_index("id", inplace=True)
        components[component]["tags"] = components[component]["tags"].apply(json.loads)

    ### Electricity transmission lines
    lines_electricity_transmission = components["lines_electricity_transmission"].copy()
    lines_electricity_transmission = _map_to_closest_region(
        lines_electricity_transmission,
        regions_onshore,
        max_distance=OFFSHORE_BUS_RADIUS,
    )
    lines_electricity_transmission = _drop_redundant_lines_links(
        lines_electricity_transmission
    )
    lines_electricity_transmission = _add_geometry_to_tags(
        lines_electricity_transmission
    )
    lines_electricity_transmission = _simplify_lines_to_380(
        lines_electricity_transmission, linetype_380
    )
    lines_electricity_transmission = _set_unique_index(lines_electricity_transmission)

    ### Electricity transmission links
    links_electricity_transmission = components["links_electricity_transmission"].copy()
    links_electricity_transmission = _map_to_closest_region(
        links_electricity_transmission,
        regions_onshore,
        max_distance=OFFSHORE_BUS_RADIUS,
    )
    links_electricity_transmission = _drop_redundant_lines_links(
        links_electricity_transmission
    )
    links_electricity_transmission = _add_geometry_to_tags(
        links_electricity_transmission
    )
    links_electricity_transmission = _set_underwater_fraction(
        links_electricity_transmission, regions_offshore
    )
    lines_electricity_transmission = _set_unique_index(lines_electricity_transmission)

    ### Hydrogen pipelines (links)
    links_hydrogen_pipeline = components["links_hydrogen_pipeline"].copy()

    ## Here
    buses_hydrogen_offshore = _create_new_buses(
        links_hydrogen_pipeline, regions_onshore, scope, "H2"
    )
    buses_hydrogen_offshore = _cluster_close_buses(
        buses_hydrogen_offshore, tol=CLUSTER_TOL
    )

    # gpd.GeoDataFrame(geometry=buses_hydrogen_offshore.apply(lambda row: Point(row["x"], row["y"]), axis=1), crs=GEO_CRS).explore(m=map, color="purple")

    links_hydrogen_pipeline = _split_to_overpassing_segments(
        links_hydrogen_pipeline,
        regions_onshore,
    )

    links_hydrogen_pipeline = _map_to_closest_region(
        links_hydrogen_pipeline,
        buses_hydrogen_offshore,
        max_distance=OFFSHORE_BUS_RADIUS,
    )
    links_hydrogen_pipeline = _map_to_closest_region(
        links_hydrogen_pipeline,
        regions_onshore,
        max_distance=OFFSHORE_BUS_RADIUS,
        add_suffix="H2",
    )
    links_hydrogen_pipeline = _drop_redundant_lines_links(links_hydrogen_pipeline)
    links_hydrogen_pipeline = _add_geometry_to_tags(links_hydrogen_pipeline)
    links_hydrogen_pipeline = _set_underwater_fraction(
        links_hydrogen_pipeline, regions_offshore
    )
    links_hydrogen_pipeline = _set_unique_index(links_hydrogen_pipeline)

    ### CO2 pipelines (links)
    links_co2_pipeline = components["links_co2_pipeline"].copy()

    buses_co2_offshore = _create_new_buses(
        links_co2_pipeline, regions_onshore, scope, "CO2", tol=OFFSHORE_BUS_RADIUS
    )
    buses_co2_offshore = _cluster_close_buses(buses_co2_offshore, tol=CLUSTER_TOL)

    links_co2_pipeline = _split_to_overpassing_segments(
        links_co2_pipeline,
        regions_onshore,
    )

    links_co2_pipeline = _map_to_closest_region(
        links_co2_pipeline,
        buses_co2_offshore,
        max_distance=OFFSHORE_BUS_RADIUS,
        add_suffix="",
    )

    links_co2_pipeline = _map_to_closest_region(
        links_co2_pipeline,
        regions_onshore,
        max_distance=OFFSHORE_BUS_RADIUS * 4,
        add_suffix="CO2",
    )

    links_co2_pipeline = _drop_redundant_lines_links(links_co2_pipeline)
    links_co2_pipeline = _add_geometry_to_tags(links_co2_pipeline)
    links_co2_pipeline = _set_underwater_fraction(links_co2_pipeline, regions_offshore)
    links_co2_pipeline = _set_unique_index(links_co2_pipeline)

    ### Map stores and storage units
    components["storage_units_hydrogen"] = _map_points_to_closest_region(
        components["storage_units_hydrogen"],
        regions_onshore,
        max_distance=OFFSHORE_BUS_RADIUS,
    )

    components["stores_co2"] = _map_points_to_closest_region(
        components["stores_co2"],
        buses_co2_offshore,
        max_distance=CLUSTER_TOL * 2,
    )
    components["stores_co2"] = _map_points_to_closest_region(
        components["stores_co2"],
        regions_onshore,
        max_distance=CLUSTER_TOL * 2,
        add_suffix="CO2",
    )

    ## DEBUG
    map = None
    map = regions_onshore.explore(m=map, color="grey")
    map = buses_hydrogen_offshore.explore(m=map)
    map = links_hydrogen_pipeline[["bus0", "bus1", "geometry"]].explore(
        m=map, color="red"
    )
    map = components["storage_units_hydrogen"].explore(m=map, color="green")
    map

    map = None
    map = regions_onshore.explore(m=map, color="grey")
    map = buses_co2_offshore.explore(m=map)
    map = links_co2_pipeline[["bus0", "bus1", "geometry"]].explore(m=map, color="red")
    map = components["stores_co2"].explore(m=map, color="green")
    map

    # Recalculate length/distances if activated
    if haversine_distance:
        n.add(
            "Bus",
            buses_hydrogen_offshore.index,
            **buses_hydrogen_offshore.drop(columns="geometry"),
        )

        n.add(
            "Bus",
            buses_co2_offshore.index,
            **buses_co2_offshore.drop(columns="geometry"),
        )

        logger.info("Recalculating line lengths with haversine distance.")
        lines_electricity_transmission = _calculate_haversine_distance(
            n, lines_electricity_transmission, line_length_factor
        )
        links_electricity_transmission = _calculate_haversine_distance(
            n, links_electricity_transmission, line_length_factor
        )
        links_hydrogen_pipeline = _calculate_haversine_distance(
            n, links_hydrogen_pipeline, line_length_factor
        )

    ### EXPORT
    ### Buses
    buses_pci_pmi_offshore = pd.concat([buses_hydrogen_offshore])
    logger.info("Exporting added PCI/PMI buses to resources folder.")
    buses_pci_pmi_offshore.to_csv(snakemake.output.buses_pci_pmi_offshore, index=True)

    ### Projects
    projects = sorted(
        lines_electricity_transmission["tags"].apply(lambda x: x["pci_code"]).unique()
    )
    logger.info("Exporting PCI/PMI projects to resources folder.")
    projects = sorted(
        lines_electricity_transmission["tags"].apply(lambda x: x["pci_code"]).unique()
    )
    logger.info(f" - Electricity transmission line (AC) projects: {projects}")
    lines_electricity_transmission[COLUMNS_LINES].to_csv(
        snakemake.output.lines_electricity_transmission, index=True
    )
    projects = sorted(
        links_electricity_transmission["tags"].apply(lambda x: x["pci_code"]).unique()
    )
    logger.info(f" - Electricity transmission links (DC) projects: {projects}")
    links_electricity_transmission[COLUMNS_LINKS].to_csv(
        snakemake.output.links_electricity_transmission, index=True
    )
    projects = sorted(
        links_hydrogen_pipeline["tags"].apply(lambda x: x["pci_code"]).unique()
    )
    logger.info(f" - Hydrogen pipeline projects: {projects}")
    links_hydrogen_pipeline[COLUMNS_LINKS].to_csv(
        snakemake.output.links_hydrogen_pipeline, index=True
    )
