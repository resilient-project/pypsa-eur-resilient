# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2021-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Clean PCI and PMI projects from interactive PCI-PMI Transparency Platform,
based on their respective types.

https://ec.europa.eu/energy/infrastructure/transparency_platform/map-viewer/main.html
"""

import json
import logging

import geopandas as gpd
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
LAYER_MAPPING = {
    "Baltic synchronisation": "links_electricity_transmission",
    "CO2 injection and surface facilities": "stores_co2_sequestration",
    "CO2 liquefaction and buffer storage": "storage_units_co2_liquefaction",
    "CO2 pipeline": "links_co2_pipeline",
    "CO2 shipping route": "links_co2_shipping",
    "Electricity line": "lines_electricity_transmission",  # contains both onshore and offshore projects, split in import; contains links and lines, split later
    "Electricity storage": "storage_units_electricity",
    "Electricity substation": "buses_electricity_transmission",  # contains both onshore and offshore projects, split in import
    "Electrolyser": "links_electrolyser",
    "Gas pipeline": "links_gas_pipeline",
    "Hydrogen pipeline": "links_hydrogen_pipeline",
    "Hydrogen storage": "storage_units_hydrogen",
    "Hydrogen terminal": "generators_hydrogen_terminal",
    "Offshore grids": "links_offshore_grids",
    "Other essential CO2 equipement": "other",  # arbitrary assignment to mark for dropping later
    "Other hydrogen assets": "other",  # arbitrary assignment to mark for dropping later
    "Smart electricity grids": "lines_smart_electricity_transmission",
    "Smart electricity grids substation": "buses_smart_electricity_transmission",
}

RENAME_COLUMNS = {
    "PCI_CODE": "pci_code",
    "COMMISSIONING_DATE": "build_year",
    "CORRIDOR_CODE": "corridor",
    "PROJECT_FICHE_SHORT_TITLE": "short_title",
    "PROJECT_FULL_TITLE": "full_title",
    "TECHNICAL_DESCR": "description",
    "IMPLEMENTATION_STATUS_DESCR": "project_status",
    "COUNTRY_CONCERNED": "countries",
    "PROMOTERS": "promoters",
    "CEF_ACTION_FICHES": "project_sheet",
    "STUDIES_OR_WORKS": "studies_works",
    "OBJECTID": "object_id",
    "layerName": "layer_name",
    "geometry": "geometry",
    "geometryType": "geometry_type",
}

COLUMNS_BUSES = [
    "project_status",
    "build_year",
    "dc",
    "v_nom",
    "tags",
    "x",
    "y",
    "geometry",
]
COLUMNS_GENERATORS = [
    "project_status",
    "build_year",
    "p_nom",
    "carrier",
    "tags",
    "x",
    "y",
    "geometry",
]
COLUMNS_LINES = [
    "project_status",
    "length",
    "build_year",
    "underground",
    "v_nom",
    "num_parallel",
    "tags",
    "x0",
    "y0",
    "x1",
    "y1",
    "geometry",
]
COLUMNS_LINKS = [
    "project_status",
    "length",
    "build_year",
    "underground",
    "p_nom",
    "carrier",
    "tags",
    "x0",
    "y0",
    "x1",
    "y1",
    "geometry",
]
COLUMNS_STORAGE_UNITS = [
    "project_status",
    "build_year",
    "p_nom",
    "max_hours",
    "tags",
    "x",
    "y",
    "geometry",
]
COLUMNS_STORES = [
    "project_status",
    "build_year",
    "e_nom",
    "tags",
    "x",
    "y",
    "geometry",
]
COMPONENTS_MAPPING = {
    "buses_electricity_transmission": COLUMNS_BUSES,
    "buses_offshore_grids": COLUMNS_BUSES,
    "buses_smart_electricity_transmission": COLUMNS_BUSES,
    "generators_hydrogen_terminal": COLUMNS_GENERATORS,
    "lines_electricity_transmission": COLUMNS_LINES,
    "links_electricity_transmission": COLUMNS_LINKS,
    "links_co2_pipeline": COLUMNS_LINKS,
    "links_co2_shipping": COLUMNS_LINKS,
    "links_electricity_transmission": COLUMNS_LINKS,
    "links_electrolyser": COLUMNS_LINKS,
    "links_gas_pipeline": COLUMNS_LINKS,
    "links_hydrogen_pipeline": COLUMNS_LINKS,
    "links_offshore_grids": COLUMNS_LINKS,
    "storage_units_co2_liquefaction": COLUMNS_STORAGE_UNITS,
    "storage_units_electricity": COLUMNS_STORAGE_UNITS,
    "storage_units_hydrogen": COLUMNS_STORAGE_UNITS,
    "stores_co2_sequestration": COLUMNS_STORES,
}


def _import_projects(filepaths):
    """
    Imports project data from a list of JSON file paths and concatenates them
    into a single DataFrame.

    Parameters:
        filepaths (list): List of file paths (str) to JSON files containing project data.

    Returns:
        df_all (pd.DataFrame): A DataFrame containing consolidated project data including their original attributes, including geometry.
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

    return df_all


def _assign_project_types(df):
    # Add value of layer mapping based on layer_name to df["project_type"]
    df["project_type"] = df["layer_name"].map(LAYER_MAPPING)

    # Assign project types based on layerName
    pci_offshore_grids = df[df["layer_name"] == "Offshore grids"]["pci_code"].unique()
    bool_buses_electricity_transmission = df["layer_name"] == "Electricity substation"
    bool_lines_electricity_transmission = df["layer_name"] == "Electricity line"
    bool_offshore_grids = df["pci_code"].isin(pci_offshore_grids)
    bool_is_point = df["geometry_type"] == "esriGeometryPoint"

    # Assign project types based on boolean conditions
    df.loc[
        bool_buses_electricity_transmission & ~bool_offshore_grids, "project_type"
    ] = LAYER_MAPPING["Electricity substation"]
    df.loc[
        bool_lines_electricity_transmission & ~bool_offshore_grids, "project_type"
    ] = LAYER_MAPPING["Electricity line"]

    # Assign offshore grids project types
    bool_buses_offshore_grids = (
        bool_buses_electricity_transmission & bool_offshore_grids
    ) | (bool_is_point & bool_offshore_grids)
    bool_lines_offshore_grids = bool_offshore_grids & ~bool_is_point

    df.loc[bool_buses_offshore_grids, "project_type"] = "buses_offshore_grids"
    df.loc[bool_lines_offshore_grids, "project_type"] = "links_offshore_grids"

    # Correct project types for lines that are actually DC links
    df.loc[
        (df["project_type"] == "lines_electricity_transmission")
        & (
            df["description"].str.contains("DC")
            | df["description"].str.contains("converter")
        ),
        "project_type",
    ] = "links_electricity_transmission"

    # Remove rows with layer_name containing other
    df = df[~df["layer_name"].str.lower().str.contains("other")]

    return df


def _create_geometries(row):
    """
    Creates geometries based on the type specified in the input row.

    Parameters:
        row (pd.Series): A pandas Series containing geometry information with the following keys:
            - 'geometry_type' (str): The type of geometry ('esriGeometryPolyline', 'esriGeometryPoint', or 'esriGeometryPolygon').
            - 'geometry' (dict): A dictionary containing the geometry data:
                - For 'esriGeometryPolyline': Contains 'paths', a list of coordinate lists.
                - For 'esriGeometryPoint': Contains 'x' and 'y' coordinates.
                - For 'esriGeometryPolygon': Contains 'rings', a list of coordinate lists.

    Returns:
        shapely.geometry.base.BaseGeometry: A Shapely geometry object (MultiLineString, Point, or Polygon) based on the input geometry type.
    """
    # Handle esriGeometryPolyline (LineString)
    if row["geometry_type"] == "esriGeometryPolyline":
        lines = [LineString(path) for path in row["geometry"]["paths"]]
        row_geom = linemerge(lines)

    # Handle esriGeometryPoint (Point)
    elif row["geometry_type"] == "esriGeometryPoint":
        point = Point(row["geometry"]["x"], row["geometry"]["y"])
        row_geom = point

    # Handle esriGeometryPolygon (Polygon)
    elif row["geometry_type"] == "esriGeometryPolygon":
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
    Remove redundant components, such as entries with 'Polygon' geometries or
    entries that are already commissioned.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        df (pd.DataFrame): The cleaned DataFrame with redundant components removed.
    """

    df = df[df["geometry"].apply(lambda x: x.geom_type != "Polygon")]
    df = df[df["project_status"] != "commissioned"]

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


def _clean_status(row):
    """
    Standardises the 'project_status' field in a given row of data.

    Parameters:
        row (pd.Series): A row containing 'project_status' column.

    Returns:
        row (pd.Series): A row with the 'project_status' field standardised.
    """
    status_mapping = dict(
        {
            "Under consideration": "under_consideration",
            "Under consideration ": "under_consideration",
            "Planned but not yet in permitting": "in_planning",
            "Permitting": "in_permitting",
            "Under construction": "under_construction",
            "Commissioned": "commissioned",
        }
    )

    if row["project_status"] in status_mapping:
        return status_mapping[row["project_status"]]

    return row["project_status"]


def _clean_columns(df):
    """
    Renames and reduces the columns in the DataFrame to only include the
    necessary columns. Further cleaning is applied to the 'build_year' (year)
    and 'project_status' (standardised) columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the original columns.

    Returns:
        df (pd.DataFrame): The cleaned DataFrame with renamed and reduced columns.
    """
    df = df[RENAME_COLUMNS.keys()].rename(columns=RENAME_COLUMNS)  # Rename columns
    df["build_year"] = df.apply(_clean_build_year, axis=1)  # Clean build_year column
    df["project_status"] = df.apply(_clean_status, axis=1)  # Clean status column

    return df


def _create_unique_ids(df):
    """
    Create unique IDs for each project, starting with the PCI code and adding a
    two-digit numerical suffix "-01", "-02", etc. only if there are multiple
    geometries for the same project.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the 'pci_code' column.

    Returns:
        df (pd.DataFrame): An indexed DataFrame with unique IDs for each project.
    """
    # Count the occurrences of each 'pci_code'
    pci_code_counts = df["pci_code"].value_counts()

    # Generate cumulative counts within each 'pci_code' group
    df["count"] = df.groupby("pci_code").cumcount() + 1  # Start counting from 1, not 0

    # Add a two-digit suffix if the 'pci_code' appears more than once
    df["suffix"] = df.apply(
        lambda row: (
            f"-{str(row['count']).zfill(2)}"
            if pci_code_counts[row["pci_code"]] > 1
            else ""
        ),
        axis=1,
    )

    # Create the 'id' column by combining 'pci_code' and suffix
    df["id"] = "PCI-" + df["pci_code"] + df["suffix"]

    # Clean up by dropping the helper columns
    df = df.drop(columns=["count", "suffix"])

    df.set_index("id", inplace=True)

    return df


def _columns_to_tags(
    df,
    columns=[
        "pci_code",
        "corridor",
        "short_title",
        "full_title",
        "description",
        "countries",
        "promoters",
        "project_sheet",
        "studies_works",
        "object_id",
        "layer_name",
        "geometry_type",
    ],
):
    """
    Converts specified columns to a 'tags' column containing a dictionary of
    tags.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the specified columns.
        columns (list): A list of column names to be converted to tags.

    Returns:
        df (pd.DataFrame): The DataFrame with the specified columns removed and a new 'tags' column containing a dictionary of tags
    """
    df["tags"] = df[columns].apply(lambda x: {k: v for k, v in zip(columns, x)}, axis=1)
    df = df.drop(columns=columns)

    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("clean_pci_pmi_projects")

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

    # INITIALISATION OF PROJECTS
    projects = _import_projects(json_files)  # Import projects from JSON files
    projects = _clean_columns(projects)
    projects = _assign_project_types(
        projects
    )  # Assign project types based on layerName

    # FIXING GEOMETRIES
    projects["geometry"] = projects.apply(
        _create_geometries, axis=1
    )  # Create Points, LineStrings, and Polygons
    projects = _remove_redundant_components(
        projects
    )  # Remove redundant components such as 'Polygon' geometries or already commissioned projects
    projects = pd.concat(
        projects.apply(_split_multilinestring, axis=1).tolist(), ignore_index=True
    )  # Split MultiLineStrings into LineStrings

    # FURTHER PROCESSING
    projects = _create_unique_ids(
        projects
    )  # Create unique IDs, dataframe is indexed by ID

    projects = _columns_to_tags(projects)

    # Create a dictionary to hold the empty DataFrames
    components = {}
    for component, columns in COMPONENTS_MAPPING.items():
        # Filter the projects DataFrame based on project_type corresponding to the component
        filtered_projects = projects[projects["project_type"] == component]

        # Create an empty DataFrame with the respective columns and the filtered index
        df = pd.DataFrame(index=filtered_projects.index, columns=columns)

        # Ensure alignment by reindexing the filtered_projects based on the empty df's index
        filtered_projects = filtered_projects.reindex(df.index)

        # Assign values from filtered_projects where the columns exist in both DataFrames
        for col in df.columns:
            if col in filtered_projects.columns:
                # Assign values using .loc to ensure correct index alignment
                df[col] = filtered_projects.loc[df.index, col]

        df = gpd.GeoDataFrame(df, crs=CRS_INPUT, geometry="geometry").to_crs(
            crs=CRS_OUTPUT
        )  # Convert to GeoDataFrame and reproject

        # Store the DataFrame in the dictionary
        components[component] = df

    # projects.tags.apply(lambda x: x.keys())

    # Export to correct output files depending on project_type
    total_count = 0
    for project_type in project_types:
        project_subset = components[project_type]
        project_count = len(
            project_subset["tags"].apply(lambda x: x["pci_code"]).unique()
        )
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
