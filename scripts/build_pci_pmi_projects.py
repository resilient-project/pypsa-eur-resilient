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
import tqdm
from _helpers import configure_logging, set_scenario_config
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

    projects = _import_projects(json_files)  # Import projects from JSON files
    projects["geometry"] = projects.apply(
        _create_geometries, axis=1
    )  # Create Points, LineStrings, and Polygons
    projects = _remove_redundant_components(
        projects
    )  # Remove redundant components such as 'Polygon' geometries

    # Split MultiLineStrings, create GeoDataFrame, and convert to WGS84 projection
    projects = pd.concat(
        projects.apply(_split_multilinestring, axis=1).tolist(), ignore_index=True
    )
    projects = gpd.GeoDataFrame(projects, crs=CRS_INPUT).to_crs(crs=CRS_OUTPUT)

    # Export to correct output files depending on project_type
    total_count = 0
    project_types = list(snakemake.output.keys())
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
