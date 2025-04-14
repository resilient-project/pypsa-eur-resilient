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

logger = logging.getLogger(__name__)


def get_costs(path):
    """
    Extracts the total system costs from the given PyPSA network.
    """
    n = pypsa.Network(path)
    
    return pd.Series({
        "capex": n.statistics.capex().sum(),
        "opex": n.statistics.opex().sum(),
    })


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_totex_heatmap",
            configfiles=["config/second-run.config.yaml"],
            )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    rule = snakemake.rule
    config = snakemake.config
    discount_rate = config["costs"]["fill_values"]["discount rate"]
    figsize = ast.literal_eval(config["plotting"][rule]["figsize"])
    fontsize = config["plotting"]["font"]["size"] 
    dpi = config["plotting"][rule]["dpi"]

    # Create df of all runs
    df_runs = pd.DataFrame()
    df_runs["path"] = snakemake.input
    df_runs["prefix"] = df_runs["path"].apply(lambda x: x.split("/")[-4])
    df_runs["name"] = df_runs["path"].apply(lambda x: x.split("/")[-3])
    df_runs["planning_horizon"] = df_runs["path"].apply(
        lambda x: x.split("/")[-1]
        .replace(".nc", "")
        .split("_")[-1]
    ).astype(int)
    
    df_runs["clusters"] = df_runs["path"].apply(
        lambda x: x.split("/")[-1]
        .replace(".nc", "")
        .split("_")[-4]
    )

    # Create data table summaries
    df_runs.loc[:, ["capex", "opex"]] = df_runs["path"].apply(get_costs)
    df_runs["totex"] = df_runs["capex"] + df_runs["opex"]

    capex = df_runs.pivot(
        index="name",
        columns="planning_horizon",
        values="capex"
    ).div(1e9) # bn EUR p.a.

    opex = df_runs.pivot(
        index="name",
        columns="planning_horizon",
        values="opex"
    ).div(1e9) # bn EUR p.a.

    totex = df_runs.pivot(
        index="name",
        columns="planning_horizon",
        values="totex"
    ).div(1e9) # bn EUR p.a.

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

    totex_pv_sum = pd.DataFrame(totex_pv.sum(axis=1), columns=["TOTEX$_{2025}$"])
    totex_pv_sum.sort_values(by="TOTEX$_{2025}$", ascending=False, inplace=True)

    # Sort the dataframes by TOTEX
    capex = capex.loc[totex_pv_sum.index]
    opex = opex.loc[totex_pv_sum.index]
    totex = totex.loc[totex_pv_sum.index]

    # Plot 
    logger.info("Plotting heatmap of total system costs.")
    plt.rc("font", **config["plotting"]["font"])
    fig = plt.figure(figsize=figsize)

    # Define grid layout with width ratios
    gs = gridspec.GridSpec(1, 4, width_ratios=[7, 7, 7, 2.5], figure=fig)

    # Create axes from the GridSpec
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)
    ax4 = fig.add_subplot(gs[3], sharey=ax1)

    for ax in [ax2, ax3, ax4]:
        ax.tick_params(left=False, labelleft=False)

    # CAPEX
    sns.heatmap(capex, annot=True, cmap="Reds", fmt=".1f", linewidths=0.5,
                cbar_kws={"label": "bn. EUR p.a."}, ax=ax1)
    ax1.set_title("CAPEX p.a.", fontsize=fontsize)
    ax1.set_xlabel("Year", fontsize=fontsize)
    ax1.set_ylabel("Scenario", fontsize=fontsize)

    # OPEX
    sns.heatmap(opex, annot=True, cmap="Blues", fmt=".1f", linewidths=0.5,
                cbar_kws={"label": "bn. EUR p.a."}, ax=ax2)
    ax2.set_title("OPEX p.a. ", fontsize=fontsize)
    ax2.set_xlabel("Year", fontsize=fontsize)
    ax2.set_ylabel("")  # Don't repeat "Scenario" if sharing y-axis

    sns.heatmap(totex, annot=True, cmap="Purples", fmt=".1f", linewidths=0.5,
                cbar_kws={"label": "bn. EUR p.a."}, ax=ax3)
    ax3.set_title("TOTEX p.a.", fontsize=fontsize)
    ax3.set_xlabel("Year", fontsize=fontsize)
    ax3.set_ylabel("")  # Don't repeat "Scenario" if sharing y-axis


    sns.heatmap(totex_pv_sum, annot=True, cmap="Greys", fmt=".0f", linewidths=0.5,
                cbar_kws={"label": "bn. EUR$_{2025}$"}, ax=ax4)
    ax4.set_title("TOTEX", fontsize=fontsize)
    ax4.set_xlabel("Present value", fontsize=fontsize)
    ax4.set_ylabel("")  # Don't repeat "Scenario" if sharing y-axis

    plt.subplots_adjust(wspace=0.1) 
    plt.savefig(snakemake.output.plot, bbox_inches="tight", dpi=dpi)