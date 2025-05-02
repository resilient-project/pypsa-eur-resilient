# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plots heatmap of total system costs for for all scenarios and planning horizons.
"""

import logging
import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import pypsa
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
            "plot_totex_heatmap",
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

    lt_order = [col for col in plotting["run_order"]]
    lt_nice_names = plotting["nice_names"]

    # Create df of all runs (rows)
    longterm = pd.DataFrame()
    longterm["path"] = snakemake.input.longterm
    longterm["prefix"] = longterm["path"].apply(lambda x: x.split("/")[-4])
    longterm["name"] = longterm["path"].apply(lambda x: x.split("/")[-3])
    longterm["lt_run"] = longterm["name"]

    index_cols = ["lt_run", "planning_horizon", "cost"]

    lt_costs = import_costs(longterm).fillna(0)
    lt_costs.set_index(index_cols, inplace=True)

    capex = (
        lt_costs.xs("capital", level="cost")["value"]
        .groupby(["lt_run", "planning_horizon"])
        .sum()
        .unstack("planning_horizon")
        .div(1e9)  # bn EUR p.a.
    )
    capex = capex.reindex(lt_order, axis=0)
    capex.index = capex.index.map(lt_nice_names)
    capex.columns = capex.columns.astype(int)

    opex = (
        lt_costs.xs("marginal", level="cost")["value"]
        .groupby(["lt_run", "planning_horizon"])
        .sum()
        .unstack("planning_horizon")
        .div(1e9)  # bn EUR p.a.
    )
    opex = opex.reindex(lt_order, axis=0)
    opex.index = opex.index.map(lt_nice_names)
    opex.columns = opex.columns.astype(int)

    totex = capex + opex

    # Create Totex timeseries, discounted
    totex_pv = pd.DataFrame(index = totex.index)
    today = 2025
    first_year = 2030
    end_year = 2060

    for year in range(first_year, end_year + 1):
        if year in range(first_year, 2040):
            totex_pv[year] = totex[2030] / ((1 + discount_rate) ** (year - today))
        if year in range(2040, 2050):
            totex_pv[year] = totex[2040] / ((1 + discount_rate) ** (year - today))
        if year in range(2050, end_year):
            totex_pv[year] = totex[2050] / ((1 + discount_rate) ** (year - today))

    totex_pv_sum = pd.DataFrame(totex_pv.sum(axis=1), columns=["NPV$_{2025}$"])
    totex_pv_sum.sort_values(by="NPV$_{2025}$", ascending=False, inplace=True)

    # Sort the dataframes by TOTEX
    capex = capex.loc[totex_pv_sum.index]
    opex = opex.loc[totex_pv_sum.index]
    totex = totex.loc[totex_pv_sum.index]

    vmin = min(capex.min().min(), opex.min().min())
    vmax = max(capex.max().max(), opex.max().max())

    # Plot 
    logger.info("Plotting heatmap of total system costs.")
    plt.rc("font", **plotting["font"])
    fig = plt.figure(figsize=figsize)

    # Define grid layout with width ratios
    gs = gridspec.GridSpec(1, 4, width_ratios=[7, 7, 7, 3.5], figure=fig)

    # Create axes from the GridSpec
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)
    ax4 = fig.add_subplot(gs[3], sharey=ax1)

    for ax in [ax2, ax3, ax4]:
        ax.tick_params(left=False, labelleft=False)

    # CAPEX
    sns.heatmap(capex, annot=True, cmap="Reds", fmt=".1f", linewidths=0.5, cbar=False, vmin = vmin, vmax = vmax,
                cbar_kws={"label": "bn. EUR p.a."}, ax=ax1, annot_kws={"fontsize": fontsize},)
    ax1.set_title("CAPEX (bn. € p.a.)", fontsize=fontsize)
    ax1.set_xlabel("", fontsize=fontsize)
    ax1.set_ylabel("Long-term scenario", fontsize=fontsize)

    # OPEX
    sns.heatmap(opex, annot=True, cmap="Reds", fmt=".1f", linewidths=0.5, cbar=False, vmin = vmin, vmax = vmax,
                cbar_kws={"label": "bn. EUR p.a."}, ax=ax2, annot_kws={"fontsize": fontsize},)
    ax2.set_title("OPEX (bn. € p.a.)", fontsize=fontsize)
    ax2.set_xlabel("Planning horizon", fontsize=fontsize)
    ax2.set_ylabel("")  # Don't repeat "Scenario" if sharing y-axis

    sns.heatmap(totex, annot=True, cmap="Purples", fmt=".1f", linewidths=0.5, cbar=False, vmin = totex.min().min(), vmax = totex.max().max(),
                cbar_kws={"label": "bn. EUR p.a."}, ax=ax3, annot_kws={"fontsize": fontsize},)
    ax3.set_title("TOTEX (bn. € p.a.)", fontsize=fontsize)
    ax3.set_xlabel("", fontsize=fontsize)
    ax3.set_ylabel("")  # Don't repeat "Scenario" if sharing y-axis

    sns.heatmap(totex_pv_sum, annot=True, cmap="Greys", fmt=".0f", linewidths=0.5, cbar=False, vmin = totex_pv_sum.min().min()*0.997, vmax=totex_pv_sum.max().max(),
                cbar_kws={"label": "bn. EUR$_{2025}$"}, ax=ax4, annot_kws={"fontsize": fontsize},)
    ax4.set_title("TOTEX (bn. €)", fontsize=fontsize)
    ax4.set_xlabel("", fontsize=fontsize)
    ax4.set_ylabel("")  # Don't repeat "Scenario" if sharing y-axis

    # Y ticks labels not rotate
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=subfontsize)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=subfontsize)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0, fontsize=subfontsize)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0, fontsize=subfontsize)
    
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=subfontsize)

    plt.subplots_adjust(wspace=0.1) 
    plt.savefig(snakemake.output.plot, bbox_inches="tight", dpi=dpi)