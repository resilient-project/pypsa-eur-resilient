# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plots regret matrix for long-term and short-term runs (columns).
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
    dpi = plotting["dpi"]

    opts = config["scenario"]["opts"][0]
    sector_opts = config["scenario"]["sector_opts"][0]

    # Create df of all runs (rows)
    df_runs_rows = pd.DataFrame()
    df_runs_rows["path"] = snakemake.input.rows
    df_runs_rows["prefix"] = df_runs_rows["path"].apply(lambda x: x.split("/")[-4])
    df_runs_rows["name"] = df_runs_rows["path"].apply(lambda x: x.split("/")[-3])
    df_runs_rows["planning_horizon"] = df_runs_rows["path"].apply(
        lambda x: x.split("/")[-1]
        .replace(".nc", "")
        .split("_")[-1]
    ).astype(int)
    
    df_runs_rows["clusters"] = df_runs_rows["path"].apply(
        lambda x: x.split("/")[-1]
        .replace(".nc", "")
        .split("_")[-4]
    )

    # Create data table summaries
    df_runs_rows.loc[:, ["capex", "opex"]] = df_runs_rows["path"].apply(get_costs)
    df_runs_rows["totex"] = df_runs_rows["capex"] + df_runs_rows["opex"]

    df_runs_rows.set_index(["name", "planning_horizon"], inplace=True)

    # List of columns
    columns = config["solve_operations"]["columns"]

    # df of secondary runs (columns)
    secondary_runs = dict()

    for col in columns:
        secondary_runs[col] = df_runs_rows.copy().reset_index()
        secondary_runs[col]["path"] = secondary_runs[col].apply(
            lambda x: "/".join(x["path"].split("/")[:-1]) + "/" + col + "/" + "base_s_ops" + "_"
            + x["clusters"] + "_" + opts + "_" + sector_opts + "_" + str(x["planning_horizon"]) + ".nc",
            axis=1
        )

        secondary_runs[col].loc[:, ["capex", "opex", "totex"]] = 0

        secondary_runs[col].loc[:, ["capex", "opex"]] = secondary_runs[col]["path"].apply(get_costs)
        secondary_runs[col]["totex"] = secondary_runs[col]["capex"] + secondary_runs[col]["opex"]

    for col in columns:
        secondary_runs[col].set_index(["name", "planning_horizon"], inplace=True)

    # Create relative deltas
    delta = dict()
    for col in columns:
        delta[col] = df_runs_rows.copy()

    numeric_cols = ["capex", "opex", "totex"]

    for col in columns:
        delta[col].loc[:, numeric_cols] = (
            secondary_runs[col].loc[:, numeric_cols]
            .subtract(df_runs_rows.loc[:, numeric_cols])
            # .divide(df_runs_rows.loc[:, numeric_cols])
            # .multiply(100)
            # .round(2)
        ).div(1e9).round(1) # bn. EUR

    # # Create long-table
    # delta_long = pd.DataFrame()
    # for col in columns:
    #     delta_long = pd.concat([delta_long, delta[col].loc[:, numeric_cols].assign(name=col)], axis=0)


    columns = [col for col in plotting["short_term_run_order"] if col in plotting["short_term_run_order"]]

    # Create heatmap of delta
    for short_term in columns:
        vmin = min(delta[short_term][numeric_cols].min().min(), vmin)
        vmax = max(delta[short_term][numeric_cols].max().max(), vmax)

    # 3x3 grid: 3 rows (short_terms), 3 columns (capex, opex, totex)
    fig = plt.figure(figsize=(figsize[0], figsize[1]*3), dpi=dpi)  # taller figure
    gs = gridspec.GridSpec(3, 3, figure=fig)

    for row_idx, short_term in enumerate(columns):
        for col_idx, col in enumerate(numeric_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            data = delta[short_term].loc[:, col].unstack().loc[plotting["run_order"]].copy()
            nice_names = plotting["nice_names"]
            data.index = data.index.map(lambda x: nice_names[x] if x in nice_names else x)

            sns.heatmap(
                data,
                annot=True,
                fmt=".1f",
                cmap="RdBu_r",
                cbar=False,
                ax=ax,
                linewidths=0.5,
                linecolor="black",    
                vmin=vmin,
                vmax=vmax,
                center=0,
            )

            ax.set_title("")
            ax.set_xlabel("")  # optional
            ax.set_ylabel("")

            if col_idx == 0:
                ax.set_ylabel(f"$\Delta$ (LT $-$ {plotting["nice_names"][short_term]})", fontsize=fontsize)
                ax.set_yticklabels(data.index, fontsize=fontsize, rotation=0)
            else:
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title(f"$\Delta$ {col.upper()} (bn. EUR)", fontsize=fontsize)

            if row_idx == 2:  # last row
                ax.set_xlabel("Planning horizon", fontsize=fontsize)

    plt.tight_layout()

    fig.savefig(
        snakemake.output[0],
        dpi=dpi,
        bbox_inches="tight",
    )