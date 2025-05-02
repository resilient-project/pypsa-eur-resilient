# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plots the PCI-PMI CO2 and H2 infrastructure on a map
"""

import logging
import ast
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from _helpers import configure_logging, set_scenario_config
from _tools import update_dict

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_pcipmi_map",
            clusters="adm",
            configfiles=["config/run5.config.yaml"],
            run="pcipmi",
            )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    rule = snakemake.rule
    config = snakemake.config
    plotting = config["plotting"]["all"]
    plotting = update_dict(plotting, snakemake.params.plotting_fig)

    figsize = ast.literal_eval(plotting["figsize"])
    fontsize = plotting["font"]["size"]
    subfontsize = fontsize-2
    titlesize = fontsize
    dpi = plotting["dpi"]

    # Read input files
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    regions_offshore = gpd.read_file(snakemake.input.regions_offshore)
    sequestration_potential = gpd.read_file(snakemake.input.sequestration_potential)
    links_co2_pipeline = gpd.read_file(snakemake.input.links_co2_pipeline)
    links_h2_pipeline = gpd.read_file(snakemake.input.links_h2_pipeline)
    stores_co2 = gpd.read_file(snakemake.input.stores_co2)
    stores_h2 = gpd.read_file(snakemake.input.stores_h2)

    alpha_regions = 0.3
    alpha_links = 0.8
    alpha_stores = 0.8
    alpha_seq = 1
    alpha_gridlines = 0.5

    # Create map
    crs = ccrs.EqualEarth()

    color_h2 = "steelblue"
    color_co2 = "indigo"
    color_seq = "darkred"

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': crs})

    plt.rc("font", **plotting["font"])

    # Add regions
    regions_onshore.to_crs(crs.proj4_init).plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.5, alpha=alpha_regions)
    regions_offshore.to_crs(crs.proj4_init).plot(ax=ax, color="lightblue", edgecolor="black", linewidth=0.5, alpha=alpha_regions)

    # Add gridlines of latitude and longitude
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), 
        draw_labels=True, 
        linewidth=0.5, 
        color='gray', 
        alpha=alpha_gridlines, 
        linestyle='--',
    )

    gl.xlabel_style = {"size": subfontsize}
    gl.ylabel_style = {"size": subfontsize}

    # Add projects
    links_co2_pipeline.to_crs(crs.proj4_init).plot(ax=ax, color=color_co2, linewidth=1, alpha=alpha_links, zorder=10)
    links_h2_pipeline.to_crs(crs.proj4_init).plot(ax=ax, color=color_h2, linewidth=1, alpha=alpha_links, zorder=10)
    stores_co2.to_crs(crs.proj4_init).plot(ax=ax, color=color_co2, edgecolor=None, linewidth=0.5, alpha=alpha_stores, zorder=20, markersize=20)
    stores_h2.to_crs(crs.proj4_init).plot(ax=ax, color=color_h2, edgecolor=None, linewidth=0.5, alpha=alpha_stores, zorder=20, markersize=20)
    sequestration_potential.to_crs(crs.proj4_init).buffer(7000).plot(ax=ax, color=color_seq, edgecolor=None, linewidth=0.5, alpha=alpha_seq, zorder=5)

    # Create a legend for the pipelines
    legend_links_co2 = plt.Line2D([0], [0], color=color_co2, linewidth=1.5, alpha=alpha_links)
    legend_links_h2 = plt.Line2D([0], [0], color=color_h2, linewidth=1.5, alpha=alpha_links)
    legend_stores_h2 = plt.Line2D([0], [0], marker="o", linewidth=0, color=color_h2, markersize=5, alpha=alpha_stores)
    legend_stores_co2 = plt.Line2D([0], [0], marker="o", linewidth=0, color=color_co2, markersize=5, alpha=alpha_stores)
    legend_seq = plt.Line2D([0], [0], marker="o", linewidth=0, color=color_seq, markersize=3, alpha=alpha_seq)

    name_links_co2 = "PCI-PMI CO$_2$ pipelines"
    name_links_h2 = "PCI-PMI H$_2$ pipelines"
    name_stores_co2 = "PCI-PMI CO$_2$ sequestration"
    name_stores_h2 = "PCI-PMI H$_2$ storages"
    name_seq = "Depleted oil & gas fields"
   
    # Add legend with border set to none
    ax.legend(
        [   
            legend_stores_co2,
            legend_stores_h2,
            legend_seq, 
            legend_links_co2, 
            legend_links_h2, 
        ], 
        [   
            name_stores_co2,
            name_stores_h2,  
            name_seq,
            name_links_co2,
            name_links_h2,    
        ], 
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        fontsize=subfontsize,
        frameon=False
    )

    # Save figure
    fig.savefig(
        snakemake.output[0],
        bbox_inches="tight",
        dpi=dpi,
    )
