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
            configfiles=["config/first-run.config.yaml"],
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

    # List of columns
    columns = config["solve_operations"]["columns"]

    # df of secondary runs (columns)
    secondary_runs = dict()

    for col in columns:
        secondary_runs[col] = df_runs_rows.copy()
        secondary_runs[col]["path"] = secondary_runs[col].apply(
            lambda x: "/".join(x["path"].split("/")[:-1]) + "/" + col + "/" + "base_s_ops" + "_"
            + x["clusters"] + "_" + opts + "_" + sector_opts + "_" + str(x["planning_horizon"]) + ".nc",
            axis=1
        )

        secondary_runs[col].loc[:, ["capex", "opex", "totex"]] = 0

        secondary_runs[col].loc[:, ["capex", "opex"]] = secondary_runs[col]["path"].apply(get_costs)
        secondary_runs[col]["totex"] = secondary_runs[col]["capex"] + secondary_runs[col]["opex"]