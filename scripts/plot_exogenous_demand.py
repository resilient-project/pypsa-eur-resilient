# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot exogenous demand for all planning_horizons.
"""

import logging
import ast
import matplotlib.pyplot as plt
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
    fig, axe = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(5.5, 5),
        dpi=dpi
    )

    # Reverse dictionary of nice_names
    nice_name_reverse = {v: k for k, v in nice_names.items()}
    nice_name_colors = {
        k: tech_colors[nice_name_reverse[k]] for k in nice_names.values()
    }

    n_df = len(planning_horizons)
    H = "/"
    labels = planning_horizons
    title = ""

    for i, ph in enumerate(planning_horizons):
        ph = str(ph)
        df = exogenous_load.query("planning_horizon==@ph").pivot(index="group", columns="nice_name", values="value")
        df = df.reindex(order)
        df_colors = [nice_name_colors.get(col, '#cccccc') for col in df.columns]

        df.plot(
            kind="bar",
            linewidth=0,
            stacked=True,
            ax=axe,
            legend=False,
            grid=False,
            color=df_colors,
        )

        n_col = len(df.columns)
        n_ind = len(df.index)

    # Adjust hatch patterns and bar spacing
    h, l = axe.get_legend_handles_labels()
    for i in range(0, n_df * n_col, n_col):
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches:
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation=90)
    axe.set_xlabel("")
    axe.set_title(title)

    # Add invisible bars for second legend (hatch)
    n = [axe.bar(0, 0, color=nice_name_colors["Biomass-based industry"], hatch=H * i * 2)[0] for i in range(n_df)]

    # First legend (colors)
    l1 = fig.legend(
        handles=h[:n_col],
        labels=l[:n_col],
        loc='lower left',
        bbox_to_anchor=(0, -0.085),
        ncol=3,
        fontsize=subfontsize,
        frameon=False,
        handleheight=1.1,
        handlelength=1,
    )

    # Second legend (hatches)
    l2 = fig.legend(
        handles=n,
        labels=labels,
        loc='upper left',
        bbox_to_anchor=(0.12, 0.98),
        ncol=1,
        fontsize=subfontsize,
        frameon=False,
        handleheight=1.1,
        handlelength=1,
    )

    # Add total value labels above bars
    bar_tops = {}
    for i in range(0, n_df * n_col, n_col):
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches:
                x_center = rect.get_x() + rect.get_width() / 2. +0.03
                height = rect.get_height()
                if height == 0:
                    continue
                if x_center in bar_tops:
                    bar_tops[x_center] += height
                else:
                    bar_tops[x_center] = height

    for x, total in bar_tops.items():
        axe.text(
            x,
            total + 150,  # adjust offset if needed
            f"{int(round(total, 0))}",
            ha='center',
            va='bottom',
            rotation=90,
            fontsize=subfontsize,
        )

    # ylim increase by factor 1.2
    axe.set_ylim(0, axe.get_ylim()[1] * 1.15)

    # xticks and yticks fontsize
    axe.tick_params(axis="x", labelsize=subfontsize)
    axe.tick_params(axis="y", labelsize=subfontsize)

    # Add y label
    axe.set_ylabel("Demand (TWh p.a.)", fontsize=fontsize)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4)

    fig.savefig(
        snakemake.output[0],
        dpi=dpi,
        bbox_inches="tight",
    )


