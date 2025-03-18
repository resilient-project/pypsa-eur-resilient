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
from shapely import segmentize, unary_union
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.ops import nearest_points

logging.getLogger("pyogrio._io").setLevel(logging.WARNING)  # disable pyogrio info
logger = logging.getLogger(__name__)

DISTANCE_CRS = "EPSG:3035"
GEO_CRS = "EPSG:4326"
OFFSHORE_BUS_RADIUS = 5000
CLUSTER_TOL = 25000
EXCLUDE_PROJECTS = ["9.8"]

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
    "e_nom",
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
    """
    Maps endpoints in a GeoDataFrame to their closest regions within a specified maximum distance.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing geometries (line geometries if lines=True, point geometries otherwise).
    regions : GeoDataFrame
        GeoDataFrame containing region geometries with a 'name' column.
    max_distance : float, optional
        Maximum allowed distance between points and regions. Points farther than this 
        will have their region set to None. Default is OFFSHORE_BUS_RADIUS.
    coords : int, optional
        Index of the coordinate to extract from line geometries when lines=True. Default is 0.
    lines : bool, optional
        Whether gdf contains line geometries. If True, points are extracted from line
        geometries using coords. If False, gdf geometries are treated as points. Default is True.
    
    Returns
    -------
    pandas.Series
        Series containing the name of the closest region for each endpoint, or None if
        the closest region is farther than max_distance.
    """
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


def _calculate_haversine_distance(buses, lines, line_length_factor):
    coords = buses[["x", "y"]]

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
    offset=0,
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
        lambda x: f"PCI-PMI {int(x.name)+1+offset}", axis=1
    )
    gdf_points.set_index("name", inplace=True)
    gdf_points["carrier"] = carrier

    return gdf_points[["x", "y", "carrier", "geometry"]]


def _split_multilinestring(row):
    """
    Splits rows containing a MultiLineString geometry into multiple rows,
    converting them to a single LineString. New rows inherit all other
    attributes from the original row. Non-MultiLineString rows are returned as-
    is.

    Parameters:
        row (pd.Series): A pandas Series containing a 'geometry' column with a MultiLineString or LineString.

    Returns:
        row (pd.Series): A row containing a LineString geometry including their original attributes.
    """
    geom = row["geometry"]
    if isinstance(geom, MultiLineString):
        # Convert MultiLineString into a list of LineStrings
        lines = [line for line in geom.geoms]
        # Create a DataFrame with the new rows, including all other columns
        return pd.DataFrame(
            {
                "geometry": lines,
                **{
                    col: [row[col]] * len(lines)
                    for col in row.index
                    if col != "geometry"
                },
            }
        )
    else:
        # Return the original row as a DataFrame, including all columns
        return pd.DataFrame([row])


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


def count_intersections(line, polygons):
    return sum(line.intersects(polygon) for polygon in polygons)


def _split_to_overpassing_segments(
    gdf, regions, distance_crs=DISTANCE_CRS, geo_crs=GEO_CRS
):
    logger.info("Splitting linestrings into segments that connect overpassing regions.")
    buffer_radius = 1 # m

    ## Delete later
    gdf_split = gdf.copy().to_crs(distance_crs)
    regions_dist = regions.to_crs(distance_crs)

    # Increase resolution of both geometries
    gdf_split["geometry"] = gdf_split["geometry"].apply(lambda x: segmentize(x, 200))
    regions_dist["geometry"] = regions_dist["geometry"].apply(
        lambda x: segmentize(x, 300)
    )

    # Do the following splitting operation only for lines that overpass multiple regions
    crosses_multiple = gdf_split.geometry.apply(lambda line: count_intersections(line, regions_dist.geometry)) > 2

    gdf_points = _find_points_on_line_overpassing_region(gdf_split.loc[crosses_multiple], regions_dist)
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



def _create_unique_ids(df):
    """
    Create unique IDs for each project, starting with the index and adding a
    two-digit numerical suffix "-01", "-02", etc. only if there are multiple
    geometries for the same project.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        df (pd.DataFrame): An indexed DataFrame with unique IDs for each project.
    """
    df = df.copy().reset_index()

    # Count the occurrences of each 'pci_code'
    counts = df["id"].value_counts()

    # Generate cumulative counts within each group
    df["count"] = df.groupby("id").cumcount() + 1  # Start counting from 1, not 0

    # Add a two-digit suffix if the id appears more than once
    df["suffix"] = df.apply(
        lambda row: (
            f"-{str(row['count']).zfill(2)}"
            if counts[row["id"]] > 1
            else ""
        ),
        axis=1,
    )

    # Create the 'id' 
    df["id"] = df["id"] + df["suffix"]

    # Clean up by dropping the helper columns
    df = df.drop(columns=["count", "suffix"])

    df.set_index("id", inplace=True)

    return df


def _cluster_close_buses(
    gdf, tol=CLUSTER_TOL, distance_crs=DISTANCE_CRS, geo_crs=GEO_CRS, offset=0
):
    carrier = gdf["carrier"].iloc[0]
    prefix = gdf.index[0].split(" ")[0]

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

    gdf["name"] = prefix + " " + (gdf.index + 1 + offset).astype(str)
    gdf["carrier"] = carrier
    gdf.set_index("name", inplace=True)

    return gdf[["x", "y", "carrier", "geometry"]]


def _aggregate_units(gdf):
    gdf = gdf.groupby(["bus", "year"]).agg(
        {
            "e_nom": "sum",
            "carrier": "first",
            "tags": lambda x: list(x.index),
            "geometry": lambda x: unary_union(x).centroid,
        }
    )

    gdf.reset_index(inplace=True)
    gdf["id"] = gdf.apply(lambda row: f"{row['bus']}", axis=1)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=GEO_CRS)

    return gdf


def _aggregate_links(gdf):

    # only keep the one with the maximum length after group
    gdf = gdf.groupby(["pci_code", "bus0", "bus1"]).agg(
        {
            "build_year": "first",
            "carrier": "first",
            "length": "max",
            "p_nom": "first",
            "tags": "first",
            "underground": "first",
            "underwater_fraction": "max",
            "geometry": "first",
        }
    ).reset_index().set_index("pci_code")

    gdf = gdf.groupby(["bus0", "bus1"]).agg(
        {
            "build_year": "max",
            "carrier": "first",
            "length": "max",
            "p_nom": "sum",
            "tags": lambda x: list(x.index),
            "underground": "first",
            "underwater_fraction": "first",
            "geometry": "first",
        }
    )

    gdf.reset_index(inplace=True)
    gdf["id"] = gdf.apply(lambda row: f"PCI-{row['tags'][0]}", axis=1)
    # Append +x if length of tags is greater than 1
    gdf["id"] = gdf.apply(
        lambda row: (
            f"{row['id']}+{len(row['tags'])-1}" if len(row["tags"]) > 1 else row["id"]
        ),
        axis=1,
    )
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=GEO_CRS)

    gdf.set_index("id", inplace=True)

    return gdf


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_pcipmi_projects",
            ll="v1.05",
            clusters="adm",
            opts="",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    line_length_factor = snakemake.params.line_length_factor
    haversine_distance = True
    linetype_380 = snakemake.config["lines"]["types"][380]

    n = pypsa.Network(snakemake.input.network)
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    regions_offshore = gpd.read_file(snakemake.input.regions_offshore).set_index("name")
    scope = gpd.read_file(snakemake.input.scope)

    buses_coords = n.buses.loc[n.buses.carrier == "AC", ["x", "y"]].copy()
    
    # Exclude projects
    exclude_projects = EXCLUDE_PROJECTS

    # H2 pipelines
    links_h2_pipeline = gpd.read_file(snakemake.input.links_h2_pipeline).set_index("id")
    links_h2_pipeline["tags"] = links_h2_pipeline["tags"].apply(json.loads)
    links_h2_pipeline["pci_code"] = links_h2_pipeline["tags"].apply(lambda x: x["pci_code"]).astype(str)
    links_h2_pipeline = links_h2_pipeline[~links_h2_pipeline["pci_code"].isin(exclude_projects)]
    
    buses_h2_offshore = _create_new_buses(
        links_h2_pipeline,
        regions_onshore,
        scope,
        "AC",
    )
    buses_h2_offshore = _cluster_close_buses(buses_h2_offshore)
    buses_coords = pd.concat([buses_coords, buses_h2_offshore[["x", "y"]]])
    links_h2_pipeline = _split_to_overpassing_segments(
        links_h2_pipeline,
        regions_onshore,
    )
    links_h2_pipeline = _map_to_closest_region(
        links_h2_pipeline,
        buses_h2_offshore,
        max_distance=OFFSHORE_BUS_RADIUS,
    )
    links_h2_pipeline = _map_to_closest_region(
        links_h2_pipeline,
        regions_onshore,
        max_distance=OFFSHORE_BUS_RADIUS,
    )
    links_h2_pipeline = _drop_redundant_lines_links(links_h2_pipeline)
    links_h2_pipeline = _add_geometry_to_tags(links_h2_pipeline)
    links_h2_pipeline = _set_underwater_fraction(
        links_h2_pipeline, 
        regions_offshore
    )
    links_h2_pipeline = _aggregate_links(links_h2_pipeline)
    links_h2_pipeline = _create_unique_ids(links_h2_pipeline)

    ### CO2 pipelines
    links_co2_pipeline = gpd.read_file(snakemake.input.links_co2_pipeline).set_index("id")
    links_co2_pipeline["tags"] = links_co2_pipeline["tags"].apply(json.loads)
    links_co2_pipeline["pci_code"] = links_co2_pipeline["tags"].apply(lambda x: x["pci_code"]).astype(str)
    links_co2_pipeline = links_co2_pipeline[~links_co2_pipeline["pci_code"].isin(exclude_projects)]
    
    buses_co2_offshore = _create_new_buses(
        links_co2_pipeline,
        regions_onshore,
        scope,
        "AC",
        offset=len(buses_h2_offshore),
    )
    buses_co2_offshore = _cluster_close_buses(
        buses_co2_offshore,
        offset=len(buses_h2_offshore),
    )
    # Append to buses_coords
    buses_coords = pd.concat([buses_coords, buses_co2_offshore[["x", "y"]]])
    links_co2_pipeline = _split_to_overpassing_segments(
        links_co2_pipeline,
        regions_onshore,
    )
    links_co2_pipeline = _map_to_closest_region(
        links_co2_pipeline,
        buses_co2_offshore,
        max_distance=OFFSHORE_BUS_RADIUS,
    )
    links_co2_pipeline = _map_to_closest_region(
        links_co2_pipeline,
        regions_onshore,
        max_distance=OFFSHORE_BUS_RADIUS * 4,
    )
    links_co2_pipeline = _drop_redundant_lines_links(links_co2_pipeline)
    links_co2_pipeline = _add_geometry_to_tags(links_co2_pipeline)
    links_co2_pipeline= _set_underwater_fraction(
        links_co2_pipeline, 
        regions_offshore,
    )
    links_co2_pipeline = _aggregate_links(links_co2_pipeline)
    links_co2_pipeline = _create_unique_ids(links_co2_pipeline)

    ### Map stores and storage units
    stores_h2 = gpd.read_file(snakemake.input.stores_h2).set_index("id")
    stores_h2["tags"] = stores_h2["tags"].apply(json.loads)
    stores_h2["pci_code"] = stores_h2["tags"].apply(lambda x: x["pci_code"]).astype(str)
    stores_h2 = stores_h2[~stores_h2["pci_code"].isin(exclude_projects)]
    # rename injection to e_nom
    stores_h2.rename(columns={"injection_withdrawal_MW": "p_nom", "storage_capacity_MWh": "e_nom"}, inplace=True)
    stores_h2 = _map_points_to_closest_region(
        stores_h2,
        regions_onshore,
        max_distance=OFFSHORE_BUS_RADIUS,
    )

    stores_co2 = gpd.read_file(snakemake.input.stores_co2).set_index("id")
    stores_co2["tags"] = stores_co2["tags"].apply(json.loads)
    stores_co2["pci_code"] = stores_co2["tags"].apply(lambda x: x["pci_code"]).astype(str)
    stores_co2 = stores_co2[~stores_co2["pci_code"].isin(exclude_projects)]
    stores_co2.rename(columns={"injection_rate_Mtpa": "e_nom"}, inplace=True)
    stores_co2["e_nom"] = stores_co2["e_nom"] * 1e6  # Convert from Mtpa to tpa
    stores_co2 = _map_points_to_closest_region(
        stores_co2,
        buses_co2_offshore,
        max_distance=CLUSTER_TOL * 2,
    )
    stores_co2 = _map_points_to_closest_region(
        stores_co2,
        regions_onshore,
        max_distance=CLUSTER_TOL * 2,
    )

    # Recalculate length/distances if activated
    if haversine_distance:
        links_h2_pipeline = _calculate_haversine_distance(
            buses_coords,
            links_h2_pipeline,
            line_length_factor,
        )
        links_co2_pipeline = _calculate_haversine_distance(
            buses_coords,
            links_co2_pipeline,
            line_length_factor,
        )

    buses_pcipmi_offshore = pd.concat([buses_h2_offshore, buses_co2_offshore])

    ### add carrier suffixes to all components
    links_h2_pipeline["bus0"] = links_h2_pipeline["bus0"] + " H2"
    links_h2_pipeline["bus1"] = links_h2_pipeline["bus1"] + " H2"
    links_co2_pipeline["bus0"] = links_co2_pipeline["bus0"] + " co2 stored"
    links_co2_pipeline["bus1"] = links_co2_pipeline["bus1"] + " co2 stored"

    stores_h2.index = stores_h2.index + " H2 Store"
    stores_h2["bus"] = stores_h2["bus"] + " H2"
    stores_co2.index = stores_co2.index + " co2 sequestered"
    stores_co2["bus"] = stores_co2["bus"] + " co2 sequestered"

    ### EXPORT
    logger.info("Exporting added PCI/PMI buses to resources folder.")
    buses_pcipmi_offshore.to_csv(snakemake.output.buses_pcipmi_offshore, index=True)

    logger.info(f" - H2 pipeline components: {len(links_h2_pipeline)}")
    links_h2_pipeline[COLUMNS_LINKS].to_csv(snakemake.output.links_h2_pipeline, index=True)
    
    logger.info(f" - CO2 pipeline components: {len(links_co2_pipeline)}")
    links_co2_pipeline[COLUMNS_LINKS].to_csv(snakemake.output.links_co2_pipeline, index=True)

    ### Export storages and stores
    logger.info("Exporting storage units and stores to resources folder.")
    logger.info(f" - H2 stores: {len(stores_h2)}")
    stores_h2[COLUMNS_STORES].to_csv(snakemake.output.stores_h2, index=True)
    logger.info(f" - CO2 stores: {len(stores_co2)}")
    stores_co2[COLUMNS_STORES].to_csv(snakemake.output.stores_co2, index=True)
