# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot costs for all scenarios side-by-side for certain carrier.
"""

import logging
import ast
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _helpers import configure_logging, set_scenario_config
from _tools import update_dict

logger = logging.getLogger(__name__)


def import_csvs(
    df: pd.DataFrame,
    data_col: str | None = None,
    index_col: list[int] = list(range(3)),
) -> pd.DataFrame:
    """
    Import costs from long-term and short-term runs.
    """

    if data_col is None:
        data_col = []
    if isinstance(data_col, str):
        data_col = [data_col]

    static_cols = ["component", "carrier"]

    data_list = []
    for i, path in enumerate(df["path"]):
        data = pd.read_csv(
            path, index_col=index_col, header=list(range(3))
        )
        # Rename three columns to
        data.columns = data.columns.get_level_values('planning_horizon')
        planning_horizons = data.columns

        data.reset_index(inplace=True)
        data = data.melt(
            id_vars=data_col + [c for c in static_cols if c in data.columns],
            value_vars=planning_horizons,
            var_name="planning_horizon",
            value_name="value",
        )
        data["name"] = df["name"].iloc[i]
        data["run"] = df["run"].iloc[i]

        data["planning_horizon"] = data["planning_horizon"].astype(str)

        # Append to cost
        data_list.append(data)
    
    data = pd.concat(data_list)

    return data


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_capacities_costs_overview_sensitivity",
            configfiles=["config/postprocess_sensitivities.config.yaml"],
            sensitivity="wy2010",
            )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config
    plotting = config["plotting"]["all"]
    plotting = update_dict(plotting, snakemake.params.plotting_fig)
    nice_names = config["plotting"]["nice_names"]
    tech_colors = config["plotting"]["tech_colors"]
    sensitivity = snakemake.wildcards.sensitivity

    discount_rate = config["costs"]["fill_values"]["discount rate"]
    figsize = ast.literal_eval(plotting["figsize"])
    fontsize = plotting["font"]["size"]
    subfontsize = fontsize
    dpi = plotting["dpi"]

    opts = config["scenario"]["opts"][0]
    sector_opts = config["scenario"]["sector_opts"][0]
    font = plotting["font"]
    legend_order = plotting["legend_order"]

    # Drop load shedding if in legend_order
    if "Load shedding" in legend_order:
        legend_order.remove("Load shedding")

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    order = [col for col in plotting["run_order"]]
    order_nice_names = [
        plotting["nice_names"][col] for col in plotting["run_order"]
    ]

    carrier_groups = config["grouping"]
    group_colors = config["group_colors"]


    # Create df of all capacities (rows)
    capacities = pd.DataFrame()
    capacities["path"] = snakemake.input.capacities
    capacities["prefix"] = capacities["path"].apply(lambda x: x.split("/")[-4])
    capacities["run"] = capacities["path"].apply(lambda x: x.split("/")[-3])

    # Split strings by __
    capacities["name"] = capacities["run"].apply(lambda x: x.split("___")[0])
    capacities["run"] = capacities["run"].apply(lambda x: x.split("___")[-1])

    # Rename main costs
    capacities.loc[capacities.prefix=="pcipmi", "name"] = "main"

    capacities = import_csvs(capacities, index_col=list(range(2))).fillna(0)
    capacities["group"] = capacities["carrier"].map(carrier_groups)
    capacities["group_color"] = capacities["group"].map(group_colors)

    # Drop stores and storage units and lines
    capacities = capacities[~capacities["component"].isin(["Store", "Line"])]

    # For all generators, remove due to double counting in links
    negative_generators = [
        "biogas", # double-check
        "coal", 
        "gas", 
        "lignite", 
        "nuclear", 
        "oil-primary", 
        "rural heat vent", 
        "solid biomass", 
        "unsustainable biogas", 
        "unsustainable bioliquids", 
        "unsustainable solid biomass",
        "urban central heat vent",
        "urban decentral heat vent",
        "load", # double-check
    ]
    capacities = capacities[~(capacities["carrier"].isin(negative_generators) & (capacities["component"]=="Generator"))]

    negative_links = [
        "BEV charger",
        "CO2 pipeline",
        "DC",
        "H2 pipeline",
        "V2G", # optional
        "agriculture machinery oil",
        "battery charger",
        "co2 sequestered",
        "coal for industry",
        "electricity distribution grid",
        "gas for industry",
        "home battery charger",
        "industry methanol",
        "kerosene for aviation",
        "land transport oil",
        "naphtha for industry",
        "oil refining",
        "process emissions",
        "rural water tanks charger",
        "shipping methanol",
        "shipping oil",
        "solid biomass for industry",
        "unsustainable bioliquids",
        "urban central water pits charger",
        "urban central water tanks charger",
        "urban decentral water tanks charger",
        'urban central water pits discharger',
        'urban central water tanks discharger',
    ]
    capacities = capacities[~(capacities["carrier"].isin(negative_links) & (capacities["component"]=="Link"))]

    # TODO: create PR on this so it's in the responsibility of the technology developer

    
    # to_drop = capacities.index[(capacities.value.abs()<10)] # Drop small values
    # capacities = capacities.drop(to_drop, axis=0)
    # capacities.reset_index(drop=True, inplace=True)

    # Group by group
    capacities = capacities.groupby(["planning_horizon", "run", "group", "name", "group_color"], observed=True).agg(
        value=("value", "sum"),
    ).div(1e3) # MW to GW
    capacities.reset_index(inplace=True)

    # Nice names for lt_run
    capacities["run"] = capacities["run"].map(plotting["nice_names"])

    capacities = capacities.sort_values(by=["planning_horizon", "run", "group"]).reset_index(drop=True)    

    # Move name column values to columns
    capacities = capacities.pivot(
        index=["planning_horizon", "run", "group", "group_color"],
        columns="name",
        values="value",
    ).reset_index()

    # Create df of all costs (rows)
    costs = pd.DataFrame()
    costs["path"] = snakemake.input.costs
    costs["prefix"] = costs["path"].apply(lambda x: x.split("/")[-4])
    costs["run"] = costs["path"].apply(lambda x: x.split("/")[-3])

    # Split strings by __
    costs["name"] = costs["run"].apply(lambda x: x.split("___")[0])
    costs["run"] = costs["run"].apply(lambda x: x.split("___")[-1])
    
    # Rename main costs
    costs.loc[costs.prefix=="pcipmi", "name"] = "main"

    # Store list of sensitivities
    sensitivities = sorted(costs.query("prefix=='sensitivities'")["name"].unique())

    costs = import_csvs(costs, data_col="cost", index_col=list(range(3))).fillna(0)
    costs["group"] = costs["carrier"].map(carrier_groups)
    costs["group_color"] = costs["group"].map(group_colors)

    # Group by group
    costs = costs.groupby(["planning_horizon", "run", "group", "name", "group_color"], observed=True).agg(
        value=("value", "sum"),
    ).div(1e9) # EUR to bn. EUR p.a.
    costs.reset_index(inplace=True)
    costs["run"] = costs["run"].map(plotting["nice_names"])

    costs = costs.sort_values(by=["planning_horizon", "run", "group"]).reset_index(drop=True)    

    # Move name column values to columns
    costs = costs.pivot(
        index=["planning_horizon", "run", "group", "group_color"],
        columns="name",
        values="value",
    ).reset_index()

    # Fill NAs with 0
    costs = costs.fillna(0)


    ### BRANCH VOLUMES
    branch_volumes = pd.DataFrame()
    branch_volumes["path"] = snakemake.input.branch_volumes
    branch_volumes["prefix"] = branch_volumes["path"].apply(lambda x: x.split("/")[-4])
    branch_volumes["run"] = branch_volumes["path"].apply(lambda x: x.split("/")[-3])

    # Split strings by __
    branch_volumes["name"] = branch_volumes["run"].apply(lambda x: x.split("___")[0])
    branch_volumes["run"] = branch_volumes["run"].apply(lambda x: x.split("___")[-1])

    # Rename main costs
    branch_volumes.loc[branch_volumes.prefix=="pcipmi", "name"] = "main"

    branch_volumes = import_csvs(branch_volumes, index_col=list(range(1))).fillna(0)
    branch_volumes["group"] = branch_volumes["carrier"].map(carrier_groups)
    branch_volumes["group_color"] = branch_volumes["group"].map(group_colors)

    # Drop AC and DC carrier
    branch_volumes = branch_volumes[~branch_volumes["carrier"].isin(["AC", "DC"])]

    branch_volumes = branch_volumes.groupby(["planning_horizon", "run", "group", "name", "group_color"], observed=True).agg(
        value=("value", "sum"),
    ).div(1e6) # MWkm to TWkm and tCO2/h*km to MtCO2/h*km
    branch_volumes.reset_index(inplace=True)
    branch_volumes["run"] = branch_volumes["run"].map(plotting["nice_names"])
    branch_volumes = branch_volumes.sort_values(by=["planning_horizon", "run", "group"]).reset_index(drop=True)

    branch_volumes = branch_volumes.pivot(
        index=["planning_horizon", "run", "group", "group_color"],
        columns="name",
        values="value",
    ).reset_index()


    x_anchor = 0
    ncol = 4
    handlelength = 1
    handleheight = 1.1

    xpad = 0.03

    toggle_labels = True
    row0_axes2 = []
    row1_axes2 = []
    row2_axes2 = []


    fig, axes = plt.subplots(
        nrows=3,
        ncols=len(planning_horizons),
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey=False, 
        tight_layout=True,
        gridspec_kw={"height_ratios": [2, 2, 1.6]},
    )
    plt.rc("font", **font)

    for a, ar in enumerate(axes):
        if a == 0:
            df = costs.copy()
            print("Plotting costs...")
        if a == 1:
            df = capacities.copy()
            print("Plotting capacities...")
        if a == 2:
            df = branch_volumes.copy()
            print("Plotting branch volumes...")

        main_totals = df[["planning_horizon", "run", "main"]].groupby(["planning_horizon", "run"]).sum()

        delta_to_main = df.copy()
        delta_to_main[sensitivities] = delta_to_main[sensitivities].subtract(delta_to_main["main"], axis=0)

        delta_abs_max = delta_to_main.groupby(["planning_horizon", "run"]).sum()[sensitivity].abs().max()

        # Drop load shedding after debugging
        if "Load shedding" in delta_to_main.group.values:
            delta_to_main = delta_to_main[delta_to_main["group"] != "Load shedding"]

        row2_axes2 = []
        for i, planning_horizon in enumerate(planning_horizons):
            ax = ar[i]
            planning_horizon = str(planning_horizon)
            data = delta_to_main.query("planning_horizon == @planning_horizon").copy().pivot(
                index="run",
                columns="group",
                values=sensitivity,
            )

            # Relative changes
            # if a == 2:
            #     data_main = delta_to_main.query("planning_horizon == @planning_horizon").copy().pivot(
            #         index="run",
            #         columns="group",
            #         values="main",
            #     )
            #     data = data.divide(data_main).fillna(0)*100

            data_order = [col for col in legend_order if col in data.columns]
            data = data[data_order]

            data = data.reindex(order_nice_names)

            if a == 0 or a == 1:
                data.plot(
                    kind="bar",
                    stacked=True,
                    ax=ax,
                    width=0.8,
                    color=[group_colors.get(col, "yellow") for col in data.columns],
                )
            
            if a == 2:
                offset = 0.21
                for i, col in enumerate(data.columns[::-1]):
                    marker = "." if "H" in col else "." if "C" in col else "o"
                    color = group_colors.get(col, "yellow")
                    x = np.arange(len(data.index)) + (i - len(data.columns)/2) * offset + offset/2

                    stem = ax.stem(
                        x,
                        data[col].values,
                        linefmt=color,
                        markerfmt=marker,
                        basefmt=" ",
                    )

                    plt.setp(stem.markerline, markersize=3, alpha=1, zorder=10)
                    plt.setp(stem.stemlines, linewidth=0.7, alpha=1, linestyle="-")

                    # if toggle_labels:
                    #     for xi, y in zip(x, data[col].values):
                    #         if abs(y) >= 0.05:
                    #             sign = "+" if y > 0 else "-"
                    #             ax.text(
                    #                 xi,
                    #                 y,
                    #                 f"{sign}{abs(y):.1f}",
                    #                 ha="center",
                    #                 va="bottom" if y >= 0 else "top",
                    #                 fontsize=subfontsize,
                    #             )


            # Turn off legend
            ax.legend().remove()

            if a == 0 or a == 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            
            if a == 0:
                ax.set_ylabel(f"$\Delta$Total system costs (bn. € p.a.)", fontsize=fontsize)

            if a == 1:
                ax.set_ylabel(f"$\Delta$Capacities (GW)", fontsize=fontsize)

            # Set title and labels
            if a == 2:
                ax.set_xlabel(f"{planning_horizon}", fontsize=fontsize)
                ax.set_xticklabels(data.index, rotation=90, fontsize=subfontsize)
                ax.set_ylabel(
                    "$\Delta$CO$_2$ pipelines (Mth$^{-1}$km)",
                    fontsize=fontsize,
                )

                if planning_horizon == "2050":
                    # right axis
                    ax2 = ax.twinx()
                    ax2.set_ylim(ax.get_ylim())
                    ax2.set_ylabel(
                        "$\Delta$H$_2$ pipelines (TWkm)",
                        fontsize=fontsize,
                    )

            # Remove all grid lines
            ax.grid(False)

            # Net totals (including negatives)
            totals = data.sum(axis=1)
            rel_totals = 100 * totals / main_totals.loc[planning_horizon].main.reindex(totals.index)

            print(rel_totals)

            # Relative change row 0, axis 2
            if a == 0:
                ax2 = ax.twinx()
                ylim_rel = [v/0.5e1 for v in ax.get_ylim()]
                ax2.set_ylim(ylim_rel)
                # Scatter all points
                ax2.scatter(
                    range(len(rel_totals)),
                    rel_totals.values,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.2,
                    s=3,
                    zorder=5,
                )
                if planning_horizon != "2050":
                    ax2.set_yticks([])
                    ax2.set_ylabel(None)
                else:
                    ax2.set_ylabel("Relative change to pathway (%)", fontsize=fontsize)

                row0_axes2.append(ax2)

            # Relative change row 1, axis 2
            if a == 1:
                ax2 = ax.twinx()
                ylim_rel = [v/0.5e2 for v in ax.get_ylim()]
                ax2.set_ylim(ylim_rel)
                # Scatter all points
                ax2.scatter(
                    range(len(rel_totals)),
                    rel_totals.values,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.2,
                    s=3,
                    zorder=5,
                )
                if planning_horizon != "2050":
                    ax2.set_yticks([])
                    ax2.set_ylabel(None)
                else:
                    ax2.set_ylabel("Relative change to pathway (%)", fontsize=fontsize)

                row1_axes2.append(ax2)


            # Only for system costs, for capacities it does not make sense
            if a == 0 or a == 1:
                # Position for label = top of positive stack
                positive_tops = data.clip(lower=0).sum(axis=1)
                for j, total in enumerate(totals):
                    if total != 0:
                        sign = "+" if total > 0 else "-"
                        fmt = ".0f" if abs(total) >= 10 else ".1f"
                        y_label = positive_tops[j] if positive_tops[j] > 0 else 0

                        if toggle_labels:
                            # main value
                            ax.text(
                                x=j,
                                y=y_label,
                                s=f"{sign}{abs(total):{fmt}}",
                                ha="center",
                                va="bottom",
                                fontsize=subfontsize,
                                zorder=20,
                            )

            # Add 0 axis line
            ax.axhline(0, color="black", lw=0.5)      


        ##### TODO indent
        # Change font size of major sharey ticks
        for ax in ar:
            ax.tick_params(axis="y", labelsize=subfontsize)

        # All borders to 0.5 thickness
        for ax in fig.get_axes():
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("black")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[c], label=c) 
        for c in legend_order[::-1] if c in list(costs.group.unique())
    ]
    handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",           # marker only
            markersize=2,           # matches s in scatter (s ≈ markersize^2)
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.2,    # matches linewidth
            label="Total change",
        )
    )

    # Add the production legend (left side, 2 columns)
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor+xpad, 0.02),  # fixed at 0 (left-aligned)
        ncol=ncol,
        fontsize=subfontsize,
        title="",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend.get_title().set_fontweight('bold')
    legend._legend_box.align = "left"    

    # Update ylims
    for idx, row in enumerate(axes):
        ymin_global = 0
        ymax_global = 0
        for col, ax in enumerate(row):
            ymin, ymax = ax.get_ylim()
            ymin_global = min(ymin_global, ymin)
            ymax_global = max(ymax_global, ymax)

            if col > 0:
                ax.yaxis.set_visible(False)

        for ax in row:
            ax.set_ylim(1.05*ymin_global, 1.13*ymax_global)
            if ax2 is not None:
                if idx == 2:
                    ax2.set_ylim(1.05*ymin_global, 1.13*ymax_global)
    
    ymin_global = 0
    ymax_global = 0
    # Update ylims of axes2s
    for ax2 in row0_axes2:
        ymin, ymax = ax2.get_ylim()
        ymin_global = min(ymin, ymin_global)
        ymax_global = max(ymax, ymax_global)

    for ax2 in row0_axes2:
        ax2.set_ylim(1.05*ymin_global, 1.13*ymax_global)
    
    ymin_global = 0
    ymax_global = 0
    for ax2 in row1_axes2:
        ymin, ymax = ax2.get_ylim()
        ymin_global = min(ymin, ymin_global)
        ymax_global = max(ymax, ymax_global)

    for ax2 in row1_axes2:
        ax2.set_ylim(1.05*ymin_global, 1.13*ymax_global)

    # Tight layout
    plt.tight_layout()

    fig.align_ylabels(fig.axes) 
    
    fig.subplots_adjust(wspace=0.05) 

    fig.savefig(
        snakemake.output[0],
        dpi=dpi,
        bbox_inches="tight",
    )
