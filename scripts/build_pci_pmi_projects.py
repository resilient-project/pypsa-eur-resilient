# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2021-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build PCI and PMI projects from interactive PCI-PMI Transparency Platform,
according to their respective types.

https://ec.europa.eu/energy/infrastructure/transparency_platform/map-viewer/main.html
"""

import json
import logging

import geopandas as gpd
import pandas as pd
from _helpers import configure_logging, set_scenario_config
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge

logger = logging.getLogger(__name__)

CRS_DATA = "EPSG:3857"  # Data is in Web Mercator projection
LAYER_MAPPING = dict(
    {
        "Baltic synchronisation": "electricity_offshore_projects",
        "CO2 injection and surface facilities": "co2_projects",
        "CO2 liquefaction and buffer storage": "co2_projects",
        "CO2 pipeline": "co2_projects",
        "CO2 shipping route": "co2_projects",
        "Electricity line": "electricity_onshore_projects",
        "Electricity storage": "electricity_onshore_projects",
        "Electricity substation": "electricity_onshore_projects",
        "Electrolyser": "electrolyser_projects",
        "Gas pipeline": "gas_projects",
        "Hydrogen pipeline": "hydrogen_projects",
        "Hydrogen storage": "hydrogen_projects",
        "Hydrogen terminal": "hydrogen_projects",
        "Offshore grids": "electricity_offshore_projects",
        "Other essential CO2 equipement": "co2_projects",
        "Other hydrogen assets": "hydrogen_projects",
        "Smart electricity grids": "smart_electricity_projects",
        "Smart electricity grids substation": "smart_electricity_projects",
    }
)


def _import_projects(filepaths):
    """
    Imports project data from a list of JSON file paths and concatenates them
    into a single DataFrame. It further adds the project type based on the
    layerName column.

    Parameters:
        filepaths (list): List of file paths (str) to JSON files containing project data.

    Returns:
        df_all (pd.DataFrame): A DataFrame containing consolidated project data including their original attributes, geometry, and project type.
    """
    df_all = pd.DataFrame()
    # Assuming list_of_projects contains the PCI codes
    for filepath in filepaths:
        with open(filepath, "r") as f:
            # Load the JSON data and append it to the list
            data = json.load(f)

        layerName = []
        attributes = []
        geometry = []
        geometryType = []

        for object in data:
            layerName.append(object["layerName"])
            attributes.append(object["attributes"])
            geometryType.append(object["geometryType"])
            geometry.append(object["geometry"])

        df = pd.DataFrame(attributes)
        df["layerName"] = layerName
        df["geometry"] = geometry
        df["geometryType"] = geometryType

        df_all = pd.concat([df_all, df], ignore_index=True)

    # add value of layer mapping based on layerName to df_all["project_type"]
    df_all["project_type"] = df_all["layerName"].map(LAYER_MAPPING)

    return df_all


def _create_geometries(row):
    """
    Creates geometries based on the type specified in the input row.

    Parameters:
        row (pd.Series): A pandas Series containing geometry information with the following keys:
            - 'geometryType' (str): The type of geometry ('esriGeometryPolyline', 'esriGeometryPoint', or 'esriGeometryPolygon').
            - 'geometry' (dict): A dictionary containing the geometry data:
                - For 'esriGeometryPolyline': Contains 'paths', a list of coordinate lists.
                - For 'esriGeometryPoint': Contains 'x' and 'y' coordinates.
                - For 'esriGeometryPolygon': Contains 'rings', a list of coordinate lists.

    Returns:
        shapely.geometry.base.BaseGeometry: A Shapely geometry object (MultiLineString, Point, or Polygon) based on the input geometry type.
    """
    # Handle polyline (LineString)
    if row["geometryType"] == "esriGeometryPolyline":
        lines = [LineString(path) for path in row["geometry"]["paths"]]

        # Create a MultiLineString directly from the list of LineString objects
        row_geom = linemerge(lines)

    # Handle point (Point)
    elif row["geometryType"] == "esriGeometryPoint":
        point = Point(row["geometry"]["x"], row["geometry"]["y"])
        row_geom = point

    # Handle polygon (Polygon)
    elif row["geometryType"] == "esriGeometryPolygon":
        for ring in row["geometry"]["rings"]:
            row_geom = Polygon(ring)

    return row_geom


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


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_pci_pmi_projects")

        # Manual updating snakemake.input for mock_snakemake, as mock_snakemake cannot yet handle input files depending on checkpoint rules
        import os

        input_dir = os.path.dirname(os.path.dirname(snakemake.input[0]))
        with open(snakemake.input[0], "r") as f:
            # Read each line, strip whitespace and newlines, and return as a list
            project_ids = [line.strip() for line in f.readlines()]
            snakemake.input = [
                f"{input_dir}/data/{project_id}.json" for project_id in project_ids
            ]

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    projects = _import_projects(snakemake.input)

    projects["geometry"] = projects.apply(_create_geometries, axis=1)
    projects = gpd.GeoDataFrame(projects, geometry="geometry", crs=CRS_DATA)
    projects = pd.concat(
        projects.apply(_split_multilinestring, axis=1).tolist(), ignore_index=True
    )
    projects = gpd.GeoDataFrame(projects, crs=CRS_DATA)
    # convert to WGS84
    projects = projects.to_crs("EPSG:4326")

    projects.explore()

    total_count = 0
    # Export to correct output files depending on project_type
    for project_type in projects.project_type.unique():
        project_subset = projects[projects.project_type == project_type]
        project_count = len(project_subset["PCI_CODE"].unique())
        logger.info(
            f"Exporting {project_count} {project_type} projects to {snakemake.output[project_type]}"
        )
        project_subset.to_file(snakemake.output[project_type], driver="GeoJSON")
        total_count += project_count

    logger.info(f"In total {total_count} projects")
    # Fix bug number of projects not correct
