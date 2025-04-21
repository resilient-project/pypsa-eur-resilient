# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plots installed capacities per run.
"""

import logging
import ast
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import seaborn as sns

from _helpers import configure_logging, set_scenario_config
from _tools import update_dict

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
            "plot_installed_capacities",
            run="greenfield-pipelines",
            configfiles=["config/dev.config.yaml"],
            )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    run = snakemake.wildcards.run
    rule = snakemake.rule
    config = snakemake.config
    plotting = config["plotting"]["all"]
    plotting = update_dict(plotting, snakemake.params.plotting_fig)

    planning_horizons = config["scenario"]["planning_horizons"]

    figsize = ast.literal_eval(plotting["figsize"])
    fontsize = plotting["font"]["size"]
    dpi = plotting["dpi"]
    grouping = config["grouping"]

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

    runs = sorted(df_runs["name"].unique())

    df = {}
    for planning_horizon in planning_horizons:
        n = pypsa.Network(
            df_runs.loc[
                (df_runs["name"] == run)
                & (df_runs["planning_horizon"] == planning_horizon),
                "path"
            ].values[0]
        )
        df[planning_horizon] = n.statistics.optimal_capacity(nice_names=False, aggregate_across_components=True)
        # TODO:fix aggregate across components (do not count all asset groupes)

    df = pd.concat(df, axis=1)
    df.fillna(0, inplace=True)

    grouped_system_costs= df.groupby(grouping).sum()

    # dummy
    subtract = pd.DataFrame(index=grouped_system_costs.index, columns=["subtract"], data=0)

    delta_system_costs= grouped_system_costs.sub(subtract["subtract"], axis=0)

    logger.info("Plotting delta system costs.")
    
    # Stacked bar plot
    norm = 1e9
    unit = "bn. € p.a."
    colors = config["group_colors"]

    # fig, ax
    fig, ax = plt.subplots(figsize=figsize)

    # Seaborn styling
    sns.set_style("white")
    sns.set_context("paper")
    plt.rc("font", **plotting["font"])

    # Normalize and sort
    plot_df = delta_system_costs
    # plot_df = plot_df[run_order]

    # Transpose for plotting (scenarios on x-axis)
    plot_df_T = plot_df.T
    totals = plot_df_T.sum(axis=0)

    # Separate into positive and negative values
    df_pos = plot_df_T.clip(lower=0)
    df_neg = plot_df_T.clip(upper=0)

    # Plot positive stacks
    bottom_pos = pd.Series(0, index=df_pos.index)
    for carrier in plot_df.index:
        values = df_pos[carrier]
        ax.bar(df_pos.index, values, bottom=bottom_pos, label=carrier, color=colors.get(carrier, None))
        bottom_pos += values

    # Plot negative stacks
    bottom_neg = pd.Series(0, index=df_neg.index)
    for carrier in plot_df.index:
        values = df_neg[carrier]
        ax.bar(df_neg.index, values, bottom=bottom_neg, label=None, color=colors.get(carrier, None))
        bottom_neg += values

    # Add totals on top
    for i, scenario in enumerate(plot_df_T.index):
        total = plot_df_T.loc[scenario].sum()
        # Format the value: add '+' if positive, otherwise show negative sign as usual
        value_text = f"{'+' if total > 0 else ''}{round(total, 1)}" 
        # Add the value text on top of the bar (or wherever you want it positioned)
        if total >= 0:
            ax.text(i,df_pos.loc[scenario].sum(), value_text, ha='center', va='bottom')
            ax.plot([i, i], [df_pos.loc[scenario].sum(), total], color='black', linestyle=':', linewidth=0.5)
        else:
            ax.text(i, df_neg.loc[scenario].sum(), value_text, ha='center', va='top')
            ax.plot([i, i], [df_neg.loc[scenario].sum(), total], color='black', linestyle=':', linewidth=0.5)
        # Add a point for each scenario
        ax.plot(i, total, marker='o', markersize=4, color='black')

        # Axis settings
        total_max = max(bottom_pos.max(), abs(bottom_neg.min()))
        ax.set_ylim(bottom_neg.min()*1.1, bottom_pos.max()*1.1)
        ax.set_xlabel("Scenarios")
        ax.set_ylabel(f"Delta system costs in {unit}")

        # Rotate x-axis labels
        ax.set_xticks(range(len(plot_df_T.index)))
        ax.set_xticklabels(plot_df_T.index, ha='right', rotation=10)

        # Add zero line
        ax.axhline(0, color='black', linewidth=0.8)

        # y axis limits to maximum of positive and negative values of total
        max_abs = max(abs(df_neg.sum(axis=1).min()), df_pos.sum(axis=1).max())
        y_buffer = 10

        ax.set_ylim(-max_abs-y_buffer, max_abs+y_buffer)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))  # remove duplicates
        ax.legend(unique.values(), unique.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.35),
                frameon=False, labelspacing=0.1, ncol=5)
        plt.setp(ax.get_legend().get_texts(), fontsize=fontsize)
        # title
        ax.set_title(planning_horizon, fontsize=fontsize)

        fig.savefig(
            snakemake.output.plot, 
            bbox_inches='tight',
            dpi=dpi,
        )