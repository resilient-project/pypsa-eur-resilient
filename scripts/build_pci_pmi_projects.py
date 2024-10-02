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

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import tqdm
from _helpers import configure_logging, set_scenario_config
from base_network import _set_links_underwater_fraction
from dateutil import parser
from pypsa.geo import haversine_pts  # to recalculate crow-flies distance
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge

logging.getLogger("pyogrio._io").setLevel(logging.WARNING)  # disable pyogrio info
logger = logging.getLogger(__name__)

CRS_DISTANCE = "EPSG:3035"

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


def _map_lines_links_to_buses(n, lines, regions_onshore):

    # only keep regions_onshore that are in the network n
    regions_onshore = regions_onshore[regions_onshore.index.isin(n.buses.index)]

    gdf_bus0 = gpd.GeoDataFrame(
        geometry=lines.apply(lambda row: Point(row["x0"], row["y0"]), axis=1),
        crs=lines.crs,
    )
    gdf_bus1 = gpd.GeoDataFrame(
        geometry=lines.apply(lambda row: Point(row["x1"], row["y1"]), axis=1),
        crs=lines.crs,
    )

    gdf_bus0 = gpd.sjoin(gdf_bus0, regions_onshore, how="left", predicate="within")
    gdf_bus1 = gpd.sjoin(gdf_bus1, regions_onshore, how="left", predicate="within")

    lines["bus0"] = gdf_bus0.loc[lines.index, "name"]
    lines["bus1"] = gdf_bus1.loc[lines.index, "name"]

    return lines


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

    # Electricity transmission lines
    lines_electricity_transmission = components["lines_electricity_transmission"].copy()
    lines_electricity_transmission = _map_lines_links_to_buses(
        n, components["lines_electricity_transmission"], regions_onshore
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

    regions_onshore_buffer = gpd.GeoDataFrame(
        regions_onshore,
        geometry=regions_onshore.to_crs("EPSG:3035").buffer(500).to_crs("EPSG:4326"),
    )

    # Electricity transmission links
    links_electricity_transmission = components["links_electricity_transmission"].copy()
    links_electricity_transmission = _map_lines_links_to_buses(
        n, components["links_electricity_transmission"], regions_onshore_buffer
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

    if haversine_distance:
        logger.info("Recalculating line lengths with haversine distance.")
        lines_electricity_transmission = _calculate_haversine_distance(
            n, lines_electricity_transmission, line_length_factor
        )
        links_electricity_transmission = _calculate_haversine_distance(
            n, links_electricity_transmission, line_length_factor
        )

    projects = sorted(
        lines_electricity_transmission["tags"].apply(lambda x: x["pci_code"]).unique()
    )

    # Export
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
