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
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
from _helpers import configure_logging, set_scenario_config
from dateutil import parser
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge

logging.getLogger("pyogrio._io").setLevel(logging.WARNING)  # disable pyogrio info
logger = logging.getLogger(__name__)

CRS_INPUT = "EPSG:3857"  # Data is in Web Mercator projection
CRS_OUTPUT = "EPSG:4326"  # Output is in WGS84 projection (as all other PyPSA-Eur data)
LAYER_MAPPING = dict(
    {
        "Baltic synchronisation": "electricity_transmission",
        "CO2 injection and surface facilities": "co2_sequestration",
        "CO2 liquefaction and buffer storage": "co2_liquefaction_storage",
        "CO2 pipeline": "co2_pipeline",
        "CO2 shipping route": "co2_shipping",
        "Electricity line": "electricity_transmission",  # contains both onshore and offshore projects, split in import
        "Electricity storage": "electricity_storage",
        "Electricity substation": "electricity_transmission",  # contains both onshore and offshore projects, split in import
        "Electrolyser": "electrolyser",
        "Gas pipeline": "gas_pipeline",
        "Hydrogen pipeline": "hydrogen_pipeline",
        "Hydrogen storage": "hydrogen_storage",
        "Hydrogen terminal": "hydrogen_terminal",
        "Offshore grids": "offshore_grids",
        "Other essential CO2 equipement": "co2_other",
        "Other hydrogen assets": "hydrogen_other",
        "Smart electricity grids": "smart_electricity_transmission",
        "Smart electricity grids substation": "smart_electricity_transmission",
    }
)
CLEAN_COLUMNS = dict(
    {
        "PCI_CODE": "pci_code",
        "COMMISSIONING_DATE": "build_year",
        "CORRIDOR_CODE": "corridor",
        "PROJECT_FICHE_SHORT_TITLE": "short_title",
        "PROJECT_FULL_TITLE": "full_title",
        "TECHNICAL_DESCR": "description",
        "IMPLEMENTATION_STATUS_DESCR": "status",
        "COUNTRY_CONCERNED": "countries",
        "PROMOTERS": "promoters",
        "CEF_ACTION_FICHES": "project_sheet",
        "STUDIES_OR_WORKS": "studies_works",
        "OBJECTID": "object_id",
        "project_type": "project_type",
        "geometry": "geometry",
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
    logger.info(f"Importing {len(filepaths)} PCI/PMI projects.")
    df_all = pd.DataFrame()
    for filepath in tqdm.tqdm(filepaths):
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

    # Add value of layer mapping based on layerName to df_all["project_type"]
    df_all["project_type"] = df_all["layerName"].map(LAYER_MAPPING)

    # Split projects of type "electricity_projects" into "electricity_onshore_projects" and "electricity_offshore_projects"
    # Logic: If the PCI_CODE appears in already "electricity_offshore_projects", their corresponding elements in "electricity_projects" need to be "electricity_offshore_projects", accordingly.
    # Otherwise, they are "electricity_onshore_projects".

    pci_offshore_grids = df_all[df_all["project_type"] == "offshore_grids"][
        "PCI_CODE"
    ].unique()
    bool_electricity_transmission = df_all["project_type"] == "electricity_transmission"
    bool_offshore_grids = df_all["PCI_CODE"].isin(pci_offshore_grids)

    df_all.loc[bool_electricity_transmission & bool_offshore_grids, "project_type"] = (
        "offshore_grids"
    )
    df_all.loc[bool_electricity_transmission & ~bool_offshore_grids, "project_type"] = (
        "electricity_transmission"
    )

    df_all = df_all[
        df_all["project_type"] != "other"
    ]  # Remove projects with project_type "other" (Components appear in other hydrogen and CO2 components)

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
    # Handle esriGeometryPolyline (LineString)
    if row["geometryType"] == "esriGeometryPolyline":
        lines = [LineString(path) for path in row["geometry"]["paths"]]
        row_geom = linemerge(lines)

    # Handle esriGeometryPoint (Point)
    elif row["geometryType"] == "esriGeometryPoint":
        point = Point(row["geometry"]["x"], row["geometry"]["y"])
        row_geom = point

    # Handle esriGeometryPolygon (Polygon)
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


def _remove_redundant_components(df):
    """
    Remove redundant components, such as entries with 'Polygon' geometries.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'geometry' column.

    Returns:
        df (pd.DataFrame): The cleaned DataFrame with redundant components removed.
    """
    df = df[df.geometry.apply(lambda x: x.geom_type != "Polygon")]

    return df


def _clean_columns(df):
    """
    Cleans columns.
    """
    df = df[CLEAN_COLUMNS.keys()].rename(columns=CLEAN_COLUMNS)

    # Clean build_year column
    df["build_year"] = df.apply(_clean_build_year, axis=1)

    return df


def _clean_build_year(row, build_year_fallback=2030):
    """
    Cleans and standardises the 'build_year' field in a given row of data.

    This function handles various cases for the 'build_year' field:
    - If 'build_year' is a string, it attempts to parse it as a date.
      - If the month is December, it returns the next year.
      - If parsing fails, it checks if the string is a 4-digit year.
      - If the string is 'Null', it uses a manual correction based on 'pci_code' or a fallback year.
    - If 'build_year' is an integer or float, it returns it as an integer.
    - If all parsing attempts fail, it returns build_year_fallback.

    Parameters:
        row (pd.Series): A row containing 'build_year' and 'pci_code' columns.

    Returns:
        year (int): A single year as an integer.
    """
    # Manual corrections: https://acer.europa.eu/sites/default/files/documents/Publications_annex/2023_ACER_PCI_Report-AnnexI_Electricity.pdf
    # PCI code
    build_year_manual = {
        "1.7.1": 2030,
        "1.7.2": 2030,
    }

    # Handle case where 'build_year' is a string
    if isinstance(row["build_year"], str):
        try:
            # Try to parse the date
            parsed_date = parser.parse(row["build_year"], dayfirst=True, fuzzy=True)
            year = parsed_date.year

            # If the month is December, return next year
            if parsed_date.month == 12:
                return year + 1
            return year
        except (ValueError, parser.ParserError):
            # Handle cases where parsing fails or format is unexpected
            if row["build_year"].isdigit() and len(row["build_year"]) == 4:
                return int(row["build_year"])
            if row["build_year"] == "Null":
                if row["pci_code"] in build_year_manual:
                    return build_year_manual[row["pci_code"]]
                return build_year_fallback

    # Handle case where 'build_year' is an integer or float
    try:
        return int(row["build_year"])
    except (ValueError, TypeError):
        return build_year_fallback


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
                f"{input_dir}/json/{project_id}.json" for project_id in project_ids
            ]

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    json_files = snakemake.input
    project_types = list(snakemake.output.keys())

    projects = _import_projects(json_files)  # Import projects from JSON files
    projects["geometry"] = projects.apply(
        _create_geometries, axis=1
    )  # Create Points, LineStrings, and Polygons

    projects = _clean_columns(projects)

    projects = _remove_redundant_components(
        projects
    )  # Remove redundant components such as 'Polygon' geometries

    # Split MultiLineStrings, create GeoDataFrame, and convert to WGS84 projection
    projects = pd.concat(
        projects.apply(_split_multilinestring, axis=1).tolist(), ignore_index=True
    )
    projects = gpd.GeoDataFrame(projects, crs=CRS_INPUT).to_crs(crs=CRS_OUTPUT)

    # for i in df
    for idx, row in df.iterrows():
        build_year = row["build_year"]
        build_year_new = row["build_year_new"]
        print(f"{build_year} -> {build_year_new}")

    project_types
    hydrogen_pipeline = projects[projects.project_type == "hydrogen_pipeline"]

    # Export to correct output files depending on project_type
    total_count = 0
    for project_type in project_types:
        project_subset = projects[projects.project_type == project_type]
        project_count = len(project_subset["PCI_CODE"].unique())
        logger.info(
            f"Exporting {project_count} {project_type} projects to {snakemake.output[project_type]}"
        )
        project_subset.to_file(snakemake.output[project_type], driver="GeoJSON")
        total_count += project_count
    logger.info(
        f"Exported a total of {total_count} projects. Note that some PCI/PMI project codes contain multiple project types."
    )


# %% Debugging
# import folium
# import branca

# colors = branca.colormap.LinearColormap(
#     colors=["red", "blue", "green", "purple", "orange", "brown"],
#     vmin=0,
#     vmax=len(project_types)-1,
# )
# colormap = {ptype: colors(i) for i, ptype in enumerate(project_types)}

# map = folium.Map(location=[50, 10], zoom_start=5)

# # Iterate over project types and add them to the map
# for i, project_type in enumerate(projects.project_type.unique()):
#     project_subset = projects[projects.project_type == project_type]

#     # Add each project's geometry to the map with customized tooltips
#     map = project_subset.explore(
#         color=colormap[project_type],
#         m=map,
#         name=project_type,
#         tooltip_kwds=dict(
#             style="max-width: 300px; word-wrap: break-word;"
#         )
#     )

# # Add the LayerControl
# folium.LayerControl(position='topright', collapsed=False).add_to(map)

# # Display the map
# map
