# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plots regret matrix for long-term and short-term runs (columns).
"""

import logging
import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _helpers import configure_logging, set_scenario_config
from _tools import update_dict

logger = logging.getLogger(__name__)


def import_costs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Import costs from long-term and short-term runs.
    """

    costs_list = []
    for i, path in enumerate(df["path"]):
        cost = pd.read_csv(
            path, index_col=list(range(3)), header=list(range(3))
        )
        # Rename three columns to
        cost.columns = cost.columns.get_level_values('planning_horizon')
        planning_horizons = cost.columns

        cost.reset_index(inplace=True)
        cost = cost.melt(
            id_vars=["cost", "component", "carrier"],
            value_vars=planning_horizons,
            var_name="planning_horizon",
            value_name="value",
        )
        cost["name"] = df["name"].iloc[i]
        cost["lt_run"] = df["lt_run"].iloc[i]

        # Append to cost
        costs_list.append(cost)
    
    costs = pd.concat(costs_list)

    return costs


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_regret_matrix",
            configfiles=["config/run5.config.yaml"],
            )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config
    plotting = config["plotting"]["all"]
    plotting = update_dict(plotting, snakemake.params.plotting_fig)

    discount_rate = config["costs"]["fill_values"]["discount rate"]
    figsize = ast.literal_eval(plotting["figsize"])
    fontsize = plotting["font"]["size"]
    subfontsize = fontsize-2
    dpi = plotting["dpi"]

    opts = config["scenario"]["opts"][0]
    sector_opts = config["scenario"]["sector_opts"][0]

    st_order = [col for col in plotting["short_term_run_order"]]


    # Create df of all runs (rows)
    longterm = pd.DataFrame()
    longterm["path"] = snakemake.input.longterm
    longterm["prefix"] = longterm["path"].apply(lambda x: x.split("/")[-4])
    longterm["name"] = longterm["path"].apply(lambda x: x.split("/")[-3])
    longterm["lt_run"] = longterm["name"]

    shortterm = pd.DataFrame()
    shortterm["path"] = snakemake.input.shortterm
    shortterm["name"] = shortterm["path"].apply(lambda x: x.split("/")[-2])
    shortterm["lt_run"] = shortterm["path"].apply(lambda x: x.split("/")[-4])

    index_cols = ["lt_run", "planning_horizon", "cost", "component", "carrier"]

    lt_costs = import_costs(longterm).fillna(0)
    lt_costs.set_index(index_cols, inplace=True)
    st_costs = import_costs(shortterm).fillna(0)
    
    st_costs = st_costs.pivot(
        index = index_cols,
        columns = "name",
        values = "value",
    ).fillna(0)

    # Calculate delta
    st_cols = st_costs.columns
    delta_costs = st_costs.copy()
    delta_costs = st_costs.subtract(lt_costs["value"], axis=0).fillna(0)

    # Plot
    delta_costs_totex = delta_costs.copy().stack().groupby(
        ["lt_run", "planning_horizon", "name"]
    ).agg(
        lambda x: x.sum()
    ).div(1e9).round(1)

    vmin = min(delta_costs_totex)
    vmax = max(delta_costs_totex)

    plt.rc("font", **plotting["font"])
    # Create heatmap with 1 row, three subfigures
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize,
        dpi=dpi,
    )


    for idx, st in enumerate(st_order):
        data = delta_costs_totex.loc[
            slice(None), slice(None), st
        ].unstack().loc[plotting["run_order"]].copy()

        nice_names = plotting["nice_names"]
        data.index = data.index.map(lambda x: nice_names[x] if x in nice_names else x)

        data_fmt = data.apply(lambda col: col.map(
            lambda x: f"+{x:.1f}" if x > 0 else (f"{x:.1f}" if x < 0 else "0")
        ))

        sns.heatmap(
            data,
            annot=data_fmt,
            fmt="",  # disable internal formatting, we're using strings
            cmap="RdBu_r",
            cbar=False,
            ax=axes[idx],
            linewidths=0.5,
            linecolor="white",
            vmin=vmin,
            vmax=vmax,
            center=0,
        )

        if idx !=0:
            axes[idx].tick_params(left=False, labelleft=False)

        axes[idx].set_title(f"$\Delta$ {plotting["nice_names"][st]}\n(bn. € p.a.)", fontsize=fontsize)
        axes[idx].set_xlabel("", fontsize=fontsize)
        axes[idx].set_ylabel("")
        axes[idx].set_yticklabels("", fontsize=subfontsize, rotation=0)
        axes[idx].tick_params(axis="x", labelsize=subfontsize)
 
    axes[0].set_yticklabels(data.index, fontsize=subfontsize, rotation=0)
    axes[0].tick_params(axis="x", labelsize=subfontsize)
    axes[1].set_xlabel("Planning horizon", fontsize=fontsize)
    axes[0].set_ylabel("Long-term scenario", fontsize=fontsize)

    fig.savefig(
        snakemake.output[0],
        dpi=dpi,
        bbox_inches="tight",
    )