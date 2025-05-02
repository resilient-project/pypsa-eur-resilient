# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create energy balance maps for the defined carriers.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from add_electricity import sanitize_carriers
from packaging.version import Version, parse
from plot_power_network import load_projection
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles, add_legend_circles
from pypsa.statistics import get_transmission_carriers
from shapely.geometry import LineString
from _tools import update_nice_names, make_square_legend_handles
import cartopy.crs as ccrs

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_balance_map",
            clusters="adm",
            opts="",
            sector_opts="",
            planning_horizons="2050",
            carrier="H2",
            configfiles=["config/run5.config.yaml"],
            run="pcipmi-national-international-expansion",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    config = snakemake.params.plotting
    font = snakemake.params.plotting_all["font"]
    carrier = snakemake.wildcards.carrier

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    n = pypsa.Network(snakemake.input.network)
    sanitize_carriers(n, snakemake.config)
    update_nice_names(n, config["nice_names"])

    n.statistics.set_parameters(round=3, drop_zero=True, nice_names=False)

    # fill empty colors or "" with light grey
    mask = n.carriers.color.isna() | n.carriers.color.eq("")
    n.carriers["color"] = n.carriers.color.mask(mask, "lightgrey")

    # set EU location with location from config
    eu_location = config["eu_node_location"]
    n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]

    # get balance map plotting parameters
    boundaries = config["map"]["boundaries"]

    config_plot = config["balance_map"][carrier]

    carrier = carrier.replace("_", " ") # TODO long-term fix
    conversion = config_plot["unit_conversion"]

    if carrier not in n.buses.carrier.unique():
        raise ValueError(
            f"Carrier {carrier} is not in the network. Remove from configuration `plotting: balance_map: bus_carriers`."
        )

    # for plotting change bus to location
    n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")

    # set location of buses to EU if location is empty and set x and y coordinates to bus location
    n.buses["x"] = n.buses.location.map(n.buses.x)
    n.buses["y"] = n.buses.location.map(n.buses.y)

    # bus_sizes according to energy balance of bus carrier
    eb = n.statistics.energy_balance(bus_carrier=carrier, groupby=["bus", "carrier"])

    # remove energy balance of transmission carriers which relate to losses
    transmission_carriers = get_transmission_carriers(n, bus_carrier=carrier).rename(
        {"name": "carrier"}
    )
    components = transmission_carriers.unique("component")
    carriers = transmission_carriers.unique("carrier")
    
    # only carriers that are also in the energy balance
    carriers_in_eb = carriers[carriers.isin(eb.index.get_level_values("carrier"))]

    eb.loc[components] = eb.loc[components].drop(index=carriers_in_eb, level="carrier")
    eb = eb.dropna()
    bus_sizes = eb.groupby(level=["bus", "carrier"]).sum().div(conversion)
    bus_sizes = bus_sizes.sort_values(ascending=False)

    colors = (
        bus_sizes.index.get_level_values("carrier")
        .unique()
        .to_series()
        .map(n.carriers.color)
    )

    # line and links widths according to optimal capacity
    flow = n.statistics.transmission(groupby=False, bus_carrier=carrier).div(conversion)

    if not flow.empty:
        flow_reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
        flow_reversed = flow[flow_reversed_mask].rename(
            lambda x: x.replace("-reversed", "")
        )
        flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)

    # if there are not lines or links for the bus carrier, use fallback for plotting
    fallback = pd.Series()
    line_widths = flow.get("Line", fallback).abs()
    link_widths = flow.get("Link", fallback).abs()

    # define maximal size of buses and branch width
    bus_size_factor = config_plot["bus_factor"]
    branch_width_factor = config_plot["branch_factor"]
    flow_size_factor = config_plot["flow_factor"]

    # get prices per region as colormap
    buses = n.buses.query("carrier in @carrier").index

    demand = n.statistics.energy_balance(bus_carrier=carrier, aggregate_time=False, groupby=["bus", "carrier"]).clip(lower=0).groupby("bus").sum().reindex(buses).rename(n.buses.location).T
    price = n.buses_t.marginal_price.reindex(buses, axis=1).rename(n.buses.location, axis=1)
    weigthed_prices=(demand*price).sum()/demand.sum()

    # Add CO2 atmosphere price
    if carrier == "co2 stored":
        emission_price = n.buses_t.marginal_price.T.loc["co2 atmosphere"].values[0]
        weigthed_prices = weigthed_prices - emission_price

    # if only one price is available, use this price for all regions
    if weigthed_prices.size == 1:
        regions["price"] = weigthed_prices.values[0]
        shift = round(weigthed_prices.values[0] / 20, 0)
    else:
        regions["price"] = weigthed_prices.reindex(regions.index).fillna(0)
        shift = 0

    link_color = config_plot.get("link_color", "darkgrey")

    vmin, vmax = regions.price.min() - shift, regions.price.max() + shift
    if config_plot["vmin"] is not None:
        vmin = config_plot["vmin"]
    if config_plot["vmax"] is not None:
        vmax = config_plot["vmax"]

    crs = load_projection(snakemake.params.plotting)

    plt.rc("font", **font)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )

    n.plot(
        bus_sizes=bus_sizes * bus_size_factor,
        bus_colors=colors,
        bus_split_circles=True,
        line_widths=line_widths * branch_width_factor,
        link_widths=link_widths * branch_width_factor,
        line_colors=link_color,
        link_colors=link_color,
        flow=flow * flow_size_factor,
        ax=ax,
        margin=0.2,
        color_geomap={"border": "darkgrey", "coastline": "darkgrey"},
        geomap=True,
        boundaries=boundaries,
    )
    
    if not link_widths.empty:
        pci_index = link_widths.loc[link_widths.index.str.contains("PCI")].index

        if not pci_index.empty:
            # Create geodataframe for PCI
            gdf_pci = pd.DataFrame(
                index=pci_index,
            )
            gdf_pci["width"] = link_widths.loc[pci_index]* branch_width_factor
            gdf_pci["width"] = gdf_pci["width"].astype(float)
            gdf_pci["x0"] = n.links.loc[pci_index].bus0.map(n.buses.x)
            gdf_pci["y0"] = n.links.loc[pci_index].bus0.map(n.buses.y)
            gdf_pci["x1"] = n.links.loc[pci_index].bus1.map(n.buses.x)
            gdf_pci["y1"] = n.links.loc[pci_index].bus1.map(n.buses.y)
            gdf_pci["geometry"] = gdf_pci.apply(
                lambda row: LineString([(row["x0"], row["y0"]), (row["x1"], row["y1"])]),
                axis=1,
            )
            gdf_pci = gpd.GeoDataFrame(gdf_pci, geometry=gdf_pci["geometry"], crs=n.crs)
            # Clip small widths
            gdf_pci = gdf_pci[gdf_pci["width"] >= 0.1]

            gdf_pci.to_crs(crs.proj4_init).plot(
                ax=ax,
                color="white",
                linewidth=gdf_pci["width"]*0.65,
                linestyle=":",
                alpha=0.8,
                zorder=4,
            )

    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="price",
        cmap=config_plot["cmap"],
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
    )

    # ax.set_title(carrier)


    # Add gridlines 
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle=':',
    )

    # Label style
    gl.xlabel_style = {"size": font["size"] - 2}
    gl.ylabel_style = {"size": font["size"] - 2}

    # Show only bottom and right labels
    gl.top_labels = False
    gl.left_labels = False
    gl.bottom_labels = True
    gl.right_labels = True

    # Move labels inside
    gl.xpadding = -1
    gl.ypadding = -1

    # Set finer gridline spacing
    gl.xlocator = plt.FixedLocator(range(-180, 181, 5))  # e.g., every 15°
    gl.ylocator = plt.FixedLocator(range(-90, 91, 5)) 

    # Add colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=config_plot["cmap"], norm=norm)
    price_unit = config_plot["region_unit"]
    cbr = fig.colorbar(
        sm,
        ax=ax,
        label=f"Demand-weighted price ({price_unit})",
        shrink=0.95,
        pad=0.015,
        aspect=50,
        orientation="horizontal",
    )
    cbr.outline.set_edgecolor("None")
    cbr.ax.tick_params(labelsize=font["size"] - 2)
    cbr.ax.xaxis.label.set_size(font["size"])

    # add legend
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
        "fontsize": font["size"] - 2,
    }

    pad = 0.12
    n.carriers.loc["", "color"] = "None"
    supp_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier").sort_values()
    cons_carriers = (
        bus_sizes[bus_sizes < 0]
        .index.unique("carrier")
        .difference(supp_carriers)
        .sort_values()
    )

    # Patch supp carriers and cons carriers
    if "H2 for industry" in supp_carriers:
        supp_carriers = supp_carriers.drop("H2 for industry")
        if "H2 for industry" not in cons_carriers:
            cons_carriers = cons_carriers.append(pd.Index(["H2 for industry"]))


    # Update sub_carriers and cons_carriers with nice_names
    supp_carriers_nn = [config["nice_names"].get(c, c) for c in supp_carriers]
    cons_carriers_nn = [config["nice_names"].get(c, c) for c in cons_carriers]

    # Sort supp and cons carriers by nice names
    supp_carriers_nn, supp_carriers = map(list, zip(
        *sorted(zip(supp_carriers_nn, supp_carriers), key=lambda x: x[0])
    ))
    cons_carriers_nn, cons_carriers = map(list, zip(
        *sorted(zip(cons_carriers_nn, cons_carriers), key=lambda x: x[0])
    ))

    x_anchor_supp = 0
    x_anchor_cons = 0.67
    ncol_supp = 2
    ncol_cons = 1
    handlelength = 1
    handleheight = 1.1

    if carrier == "H2":
        x_anchor_cons = 0.5
        ncol_cons = 2

    # Add supply carriers
    add_legend_patches(
        ax,
        n.carriers.color[supp_carriers],
        supp_carriers_nn,
        legend_kw={
            "bbox_to_anchor": (x_anchor_supp, -pad),
            "ncol": ncol_supp,
            "title": "Production",
            "handlelength": handlelength,
            "handleheight": handleheight,
            **legend_kwargs,
        },
    )

    # Add consumption carriers
    add_legend_patches(
        ax,
        n.carriers.color[cons_carriers],
        cons_carriers_nn,
        legend_kw={
            "bbox_to_anchor": (x_anchor_cons, -pad),
            "ncol": ncol_cons,
            "title": "Consumption",
            "handlelength": handlelength,
            "handleheight": handleheight,
            **legend_kwargs,
        },
    )

    # Add bus legend
    legend_bus_sizes = config_plot["bus_sizes"]
    carrier_unit = config_plot["unit"]
    if legend_bus_sizes is not None:
        add_legend_circles(
            ax,
            [
                s * bus_size_factor * 1 #SEMICIRCLE_CORRECTION_FACTOR
                for s in legend_bus_sizes
            ],
            [f"{s} {carrier_unit}" for s in legend_bus_sizes],
            patch_kw={"color": link_color},
            legend_kw={
                "bbox_to_anchor": (0.03, 0.97),
                **legend_kwargs,
            },
        )

    # Add branch legend
    legend_branch_sizes = config_plot["branch_sizes"]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax,
            [s * branch_width_factor for s in legend_branch_sizes],
            [f"{s} {carrier_unit}" for s in legend_branch_sizes],
            patch_kw= {
                "color": link_color,
                "solid_capstyle": "round",
            },
            legend_kw={
                "bbox_to_anchor": (0.21, 0.97), 
                "handlelength": 1.4,
                **legend_kwargs
            },
        )

    # Update legend entries using nice names
    handles, labels = ax.get_legend_handles_labels()
    labels_nn = [config["nice_names"].get(label, label) for label in labels]
    ax.legend(handles, labels_nn)

    fig.savefig(
        snakemake.output[0],
        dpi=400,
        bbox_inches="tight",
    )
