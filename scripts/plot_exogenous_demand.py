# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot exogenous demand for all planning_horizons.
"""

import logging
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

from _helpers import configure_logging, set_scenario_config
from _tools import update_dict

logger = logging.getLogger(__name__)


def import_csvs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Import costs from long-term and short-term runs.
    """

    data_col = "bus_carrier"

    data_list = []
    for i, path in enumerate(df["path"]):
        data = pd.read_csv(
            path, index_col=list(range(3)), header=list(range(3))
        )
        # Rename three columns to
        data.columns = data.columns.get_level_values('planning_horizon')
        planning_horizons = data.columns

        data.reset_index(inplace=True)
        data = data.melt(
            id_vars=[data_col, "component", "carrier"],
            value_vars=planning_horizons,
            var_name="planning_horizon",
            value_name="value",
        )
        data["name"] = df["name"].iloc[i]
        data["lt_run"] = df["lt_run"].iloc[i]

        data["planning_horizon"] = data["planning_horizon"].astype(str)

        # Append to cost
        data_list.append(data)
    
    data = pd.concat(data_list)

    return data


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_exogenous_demand",
            configfiles=["config/run5.config.yaml"],
            carrier="co2 stored",
            )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config
    plotting = config["plotting"]["all"]
    plotting = update_dict(plotting, snakemake.params.plotting_fig)
    nice_names = config["load_nice_names"]
    load_group = config["load_grouping"]
    tech_colors = config["plotting"]["tech_colors"]

    discount_rate = config["costs"]["fill_values"]["discount rate"]
    figsize = ast.literal_eval(plotting["figsize"])
    fontsize = plotting["font"]["size"]
    subfontsize = fontsize-2
    dpi = plotting["dpi"]

    opts = config["scenario"]["opts"][0]
    sector_opts = config["scenario"]["sector_opts"][0]
    carrier = snakemake.wildcards.carrier
    font = plotting["font"]

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    lt_order = [col for col in plotting["run_order"]]
    lt_order_nice_names = [
        plotting["nice_names"][col] for col in plotting["run_order"]
    ]

    # Create df of all runs (rows)
    longterm = pd.DataFrame()
    longterm["path"] = snakemake.input.longterm
    longterm["prefix"] = longterm["path"].apply(lambda x: x.split("/")[-4])
    longterm["name"] = "Long-term"
    longterm["lt_run"] = longterm["path"].apply(lambda x: x.split("/")[-3])

    lt_balances = import_csvs(longterm).fillna(0)
    lt_balances["run_type"] = "LT"

    index_cols = ["carrier", "planning_horizon"]
    exogenous_load = lt_balances.query("component == 'Load'").copy()

    exogenous_load = exogenous_load.groupby(
        index_cols, 
        observed=True,
    ).agg(value=("value", "mean")).reset_index()
    
    # Mapping
    exogenous_load["nice_name"] = exogenous_load["carrier"].map(nice_names)
    exogenous_load["group"] = exogenous_load["carrier"].map(load_group)

    # Make positive
    exogenous_load["value"] = exogenous_load["value"].abs()

    # Divide by 1e6 to get TWh
    exogenous_load["value"] = exogenous_load["value"].div(1e6)

    # Sort by 2050
    order = exogenous_load.query("planning_horizon == '2050'").groupby(
        "group", observed=True
    ).agg(value=("value", "sum")).sort_values(by="value", ascending=True).index

    plt.rc("font", **plotting["font"])

    # Set up figure and axis
    not_twh = ["process emissions"]
    exogenous_load_mt = exogenous_load.query("carrier in @not_twh").copy()
    exogenous_load_twh = exogenous_load.drop(
        exogenous_load.query("carrier in @not_twh").index
    ).copy()

    nice_name_reverse = {v: k for k, v in nice_names.items()}
    nice_name_colors = {
        k: tech_colors[nice_name_reverse[k]] for k in nice_names.values()
    }

        
    # === Parameters ===
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={'width_ratios': [8, 1]},  # Left wider, right narrower
        sharey=False
    )
    H = "//"  # Hatch pattern
    n_df = len(planning_horizons)  # Number of planning horizon groups

    # === Helper function ===
    def plot_grouped_stacked_bar(ax, df_all, order, ylabel, unit_gap, hatch_color):
        bar_tops = {}
        for i, ph in enumerate(planning_horizons):
            ph_str = str(ph)
            df = df_all.query("planning_horizon == @ph_str").pivot(index="group", columns="nice_name", values="value")
            df = df.reindex(order)
            colors = [nice_name_colors.get(col, '#cccccc') for col in df.columns]

            df.plot(kind="bar", stacked=True, linewidth=0, ax=ax, color=colors, legend=False)

            n_col, n_ind = len(df.columns), len(df.index)

            # Hatch and shift bars
            handles, _ = ax.get_legend_handles_labels()
            for j, pa in enumerate(handles[i * n_col:(i + 1) * n_col]):
                for rect in pa.patches:
                    rect.set_x(rect.get_x() + i / (n_df + 1))
                    rect.set_width(1 / (n_df + 1))
                    rect.set_hatch(H * i)
                    rect.set_edgecolor(hatch_color)

                    # Accumulate bar tops for value labels
                    x = rect.get_x() + rect.get_width() / 2.
                    bar_tops[x] = bar_tops.get(x, 0) + rect.get_height()

        # X ticks and labels
        ax.set_xticks(np.arange(len(order))+0.125)
        ax.set_xticklabels(order, rotation=90, fontsize=subfontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis="y", labelsize=subfontsize)
        ax.set_xlabel("")

        # Value labels
        for x, total in bar_tops.items():
            ax.text(x+0.025, total + unit_gap, f"{int(round(total))}", ha='center', va='bottom', fontsize=subfontsize, rotation=90)

        # Y axis limits 
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max*1.1)

    # === Plot TWh (left) ===
    order_twh = order[~np.isin(order, exogenous_load_mt["group"])]
    plot_grouped_stacked_bar(
        axes[0], 
        exogenous_load_twh, 
        order_twh, 
        "Demand (TWh)", 
        unit_gap=100,
        hatch_color="black",
    )

    # === Plot Mt p.a. (right) ===
    order_mt = order[order.isin(exogenous_load_mt["group"])]
    plot_grouped_stacked_bar(
        axes[1], 
        exogenous_load_mt, 
        order_mt, 
        "Demand (Mt p.a.)", 
        unit_gap=3,
        hatch_color="white",
    )

    handleheight = 1.1
    handlelength = 1

    # Legend for colors (technologies)
    color_handles = [
        Patch(
            facecolor=color, 
            edgecolor=None, 
            label=name,
            ) for name, color in nice_name_colors.items()
    ]

    # Legend for hatch patterns (planning horizons)
    hatch_handles = [
        Patch(
            facecolor=nice_name_colors["Biomass-based industry"],
            edgecolor=None,
            hatch=H * i,
            label=str(ph),
        ) for i, ph in enumerate(planning_horizons)
    ]

    fig.legend(
        handles=color_handles,
        loc='lower left',
        bbox_to_anchor=(0, -0.1),
        ncol=3,
        fontsize=subfontsize,
        frameon=False,
        handleheight=handleheight,
        handlelength=handlelength,
    )

    # Add hatch legend (planning horizons) above the plot
    fig.legend(
        handles=hatch_handles,
        loc='upper left',
        bbox_to_anchor=(0.12, 0.973),
        ncol=1,
        fontsize=subfontsize,
        frameon=False,
        handleheight=handleheight,
        handlelength=handlelength,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4)

    fig.savefig(
        snakemake.output[0],
        dpi=dpi,
        bbox_inches="tight",
    )


