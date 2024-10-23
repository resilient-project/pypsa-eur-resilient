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
from pypsa.geo import haversine_pts  # to recalculate crow-flies distance
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import snap

logging.getLogger("pyogrio._io").setLevel(logging.WARNING)  # disable pyogrio info
logger = logging.getLogger(__name__)

CRS_DISTANCE = "EPSG:3035"
CRS_WGS84 = "EPSG:4326"
MAX_DISTANCE = 3000

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


def _map_endpoints_to_closest_region(gdf, regions, max_distance=MAX_DISTANCE, coords=0):
    gdf_points = gdf.geometry.apply(lambda x: Point(x.coords[coords]))

    gdf_points = gpd.GeoDataFrame(geometry=gdf_points, crs=CRS_WGS84)
    # Spatial join nearest with regions

    # Find nearest region index
    regions = regions.to_crs(CRS_DISTANCE)
    gdf_points = gdf_points.to_crs(CRS_DISTANCE)

    gdf_points = gpd.sjoin_nearest(gdf_points, regions, how="left")
    gdf_points = gdf_points.join(
        regions, on="name", lsuffix="_point", rsuffix="_region"
    )
    gdf_points["distance"] = gdf_points.apply(
        lambda x: x.geometry_point.distance(x.geometry_region), axis=1
    )

    bool_too_far = gdf_points["distance"] > MAX_DISTANCE
    gdf_points.loc[bool_too_far, "name"] = None

    return gdf_points["name"]


def _map_to_closest_region(gdf, regions, max_distance=MAX_DISTANCE, add_suffix=None):
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
        links.intersection(regions_offshore.union_all()).to_crs(CRS_DISTANCE).length
        / links.to_crs(CRS_DISTANCE).length
    ).round(2)

    return links


def _create_new_buses(gdf, regions, scope, carrier):
    buffered_regions = (
        regions.to_crs(CRS_DISTANCE).buffer(5000).to_crs(CRS_WGS84).union_all()
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
        geometry=[geom for geom in list_points.geoms], crs=CRS_WGS84
    )

    # Extract x and y coordinates into separate columns
    gdf_points["x"] = gdf_points.geometry.x
    gdf_points["y"] = gdf_points.geometry.y
    gdf_points["name"] = gdf_points.apply(
        lambda x: f"PCI-PMI {int(x.name)+1} {carrier}", axis=1
    )
    gdf_points.set_index("name", inplace=True)
    gdf_points["carrier"] = carrier
    gdf_points["location"] = gdf_points.index

    gdf_points["geometry"] = (
        gdf_points.to_crs(CRS_DISTANCE).buffer(MAX_DISTANCE).to_crs(CRS_WGS84)
    )

    return gdf_points[["x", "y", "carrier", "location", "geometry"]]


# def _find_overpassing_regions(link, regions):
#     link["link_index"] = link.index
#     regions["region_index"] = regions.index
#     overlap = gpd.overlay(link, regions)


#     overlap["center_point"] = overlap["geometry"].apply(
#         lambda l: l.interpolate(l.length / 2)
#     )

#     overlap["on_point"] = overlap.apply(
#         lambda row: nearest_points(row["center_point"], row["geometry"])[1],
#         axis=1
#     )
#     overlap["on_point"].crs=CRS_WGS84

#     link_segments = [l for l in overlap.geometry]
#     regions = [r for r in overlap.region_index]

#     return overpassing_regions


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_pci_pmi_projects",
            ll="vopt",
            clusters=256,
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
        max_distance=MAX_DISTANCE,
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

    ### Electricity transmission links
    links_electricity_transmission = components["links_electricity_transmission"].copy()
    links_electricity_transmission = _map_to_closest_region(
        links_electricity_transmission,
        regions_onshore,
        max_distance=MAX_DISTANCE,
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

    ### Hydrogen pipelines (links)
    links_hydrogen_pipeline = components["links_hydrogen_pipeline"].copy()

    ## Here
    buses_hydrogen_offshore = _create_new_buses(
        links_hydrogen_pipeline, regions_onshore, scope, "H2"
    )

    links_hydrogen_pipeline = _map_to_closest_region(
        links_hydrogen_pipeline,
        buses_hydrogen_offshore,
        max_distance=MAX_DISTANCE,
    )
    links_hydrogen_pipeline = _map_to_closest_region(
        links_hydrogen_pipeline,
        regions_onshore,
        max_distance=MAX_DISTANCE,
        add_suffix="H2",
    )
    links_hydrogen_pipeline = _drop_redundant_lines_links(links_hydrogen_pipeline)
    links_hydrogen_pipeline = _add_geometry_to_tags(links_hydrogen_pipeline)
    links_hydrogen_pipeline = _set_underwater_fraction(
        links_hydrogen_pipeline, regions_offshore
    )
    # TODO: Split pipeline into segments, if they overpass multiple regions

    ### CO2 pipelines (links)
    # links_co2_pipeline = components["links_co2_pipeline"].copy()

    # Recalculate length/distances if activated
    if haversine_distance:
        n.madd(
            "Bus",
            buses_hydrogen_offshore.index,
            **buses_hydrogen_offshore.drop(columns="geometry"),
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
