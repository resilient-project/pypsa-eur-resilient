
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build regionalised geological sequestration potential for carbon dioxide using
data from `CO2Stop <https://setis.ec.europa.eu/european-co2-storage-
database_en>`_.
"""

import logging

import geopandas as gpd
import pandas as pd
from _helpers import configure_logging, set_scenario_config
from shapely.algorithms.polylabel import polylabel

logger = logging.getLogger(__name__)


def area(gdf):
    """
    Returns area of GeoDataFrame geometries in square kilometers.
    """
    return gdf.to_crs(epsg=3035).area.div(1e6)


def allocate_sequestration_potential(
    gdf, regions, attr="conservative estimate Mt", threshold=3
):
    if isinstance(attr, str):
        attr = [attr]
    gdf = gdf.loc[gdf[attr].sum(axis=1) > threshold, attr + ["geometry"]]
    gdf["area_sqkm"] = area(gdf)
    overlay = gpd.overlay(regions, gdf, keep_geom_type=True)
    overlay["share"] = area(overlay) / overlay["area_sqkm"]
    adjust_cols = overlay.columns.difference({"name", "area_sqkm", "geometry", "share"})
    overlay[adjust_cols] = overlay[adjust_cols].multiply(overlay["share"], axis=0)

    # Clustering
    overlay.to_crs(epsg=3035, inplace=True)
    buffer_shapes = overlay.groupby("name").agg(
        {"geometry": lambda x: x.union_all(), "area_sqkm": "sum"}
    )
    buffer_shapes = gpd.GeoDataFrame(buffer_shapes, geometry="geometry", crs=overlay.crs)
    buffer_shapes["geometry"] = buffer_shapes.apply(
        lambda x: x["geometry"].buffer(50*1e3), axis=1
    )
    # Explode
    buffer_shapes = buffer_shapes.explode()
    buffer_shapes["poi"] = buffer_shapes.apply(lambda x: polylabel(x.geometry, tolerance=10), axis=1)
    buffer_shapes["poi"].crs = buffer_shapes.crs
    buffer_shapes["id"] = buffer_shapes.groupby("name").cumcount()
    buffer_shapes.reset_index(inplace=True)
    buffer_shapes["id"] = buffer_shapes["name"] + " offshore " + buffer_shapes["id"].astype(str) if buffer_shapes["id"].max() > 0 else buffer_shapes["name"]
    buffer_shapes.set_index("name", inplace=True)

    overlay["cluster"] = overlay.apply(
        lambda x: gpd.sjoin_nearest(
            gpd.GeoDataFrame(geometry=[x.geometry], crs = overlay.crs), 
            buffer_shapes.loc[buffer_shapes.index == x["name"]],
        )["id"].values[0],
        axis=1,
    )

    overlay = overlay.groupby("cluster").agg(
        {
            "name": "first",
            "area_sqkm": "sum",
            "geometry": lambda x: x.union_all(),
            **{a: "sum" for a in attr},
        }
    )
    overlay["total_estimate_Mt"] = overlay[attr].sum(axis=1)
    overlay["total_estimate_Mt"] = overlay["total_estimate_Mt"].round(3)

    # Map PoI
    buffer_shapes.set_index("id", inplace=True)
    overlay = gpd.GeoDataFrame(overlay, geometry="geometry", crs="EPSG:3035")

    # Update area
    overlay["area_sqkm"] = area(overlay)

    buffer_shapes.to_crs(epsg=4326, inplace=True)
    buffer_shapes["poi"] = buffer_shapes["poi"].to_crs(epsg=4326)
    overlay.to_crs(epsg=4326, inplace=True)
    overlay["x"] = buffer_shapes.loc[overlay.index, "poi"].x
    overlay["y"] = buffer_shapes.loc[overlay.index, "poi"].y

    # Rename "name" to "bus_onshore"
    overlay.rename(columns={"name": "bus_onshore"}, inplace=True)

    return overlay[["bus_onshore", "total_estimate_Mt", "x", "y", "geometry"]]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_sequestration_potentials", clusters="adm")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    cf = snakemake.params.sequestration_potential

    gdf = gpd.read_file(snakemake.input.sequestration_potential)

    # Drop duplicates
    gdf.drop_duplicates(subset="ID", inplace=True)

    regions = gpd.read_file(snakemake.input.regions_offshore)
    if cf["include_onshore"]:
        onregions = gpd.read_file(snakemake.input.regions_onshore)
        regions = pd.concat([regions, onregions]).dissolve(by="name").reset_index()

    s = allocate_sequestration_potential(
        gdf, regions, attr=cf["attribute"], threshold=cf["min_size"],
    )

    s = s.where(s.total_estimate_Mt > cf["min_size"]).dropna()

    # Export
    s.to_file(snakemake.output.sequestration_potential)