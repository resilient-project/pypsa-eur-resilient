# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plots delta balances for a carrier.
"""

import logging
import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
            "plot_delta_balances",
            configfiles=["config/run5.config.yaml"],
            carrier="process emissions CC"
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
    carrier = snakemake.wildcards.carrier

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    st_order = [col for col in plotting["short_term_run_order"]]
    lt_order = [col for col in plotting["run_order"]]
    lt_order_nice_names = [
        plotting["nice_names"][col] for col in plotting["run_order"]
    ]

    # Plot additional settings
    custom_markers = plotting["custom_markers"]
    custom_hues = plotting["custom_hues"]

    # Create df of all runs (rows)
    longterm = pd.DataFrame()
    longterm["path"] = snakemake.input.longterm
    longterm["prefix"] = longterm["path"].apply(lambda x: x.split("/")[-4])
    longterm["name"] = "Long-term"
    longterm["lt_run"] = longterm["path"].apply(lambda x: x.split("/")[-3])

    shortterm = pd.DataFrame()
    shortterm["path"] = snakemake.input.shortterm
    shortterm["name"] = shortterm["path"].apply(lambda x: x.split("/")[-2])
    shortterm["lt_run"] = shortterm["path"].apply(lambda x: x.split("/")[-4])

    index_cols = ["lt_run", "planning_horizon", "bus_carrier", "component", "carrier"]

    lt_balances = import_csvs(longterm).fillna(0)
    lt_balances["run_type"] = "LT"
    st_balances = import_csvs(shortterm).fillna(0)
    st_balances["run_type"] = "ST"

    balances = pd.concat([lt_balances, st_balances], axis=0).reset_index(drop=True)
    to_drop = balances.index[(balances.value.abs()<10)] # Drop small values
    balances = balances.drop(to_drop, axis=0)
    balances.reset_index(drop=True, inplace=True)

    # Rename to nice names
    balances["name"] = balances["name"].replace(plotting["nice_names"])
    balances["lt_run"] = balances["lt_run"].replace(plotting["nice_names"])
    
    # Carrier nice_name
    bus_carrier_nice_name = {
        "AC": "Electricity",
        "co2": "CO$_2$",
        "co2 stored": "CO$_2$",
        "H2": "H$_2$",
        "process emissions": "Process emissions",
    }

    bus_carrier_selected = {
        "DAC": "co2",
        "H2 Electrolysis": "H2",
        "H2 Fuel Cell": "H2",
        "co2 sequestered": "co2 stored",
        "SMR CC": "co2 stored",
        "gas for industry CC": "co2 stored",
        "process emissions CC": "co2 stored",
        "solid biomass for industry CC": "co2 stored",
        "urban central CHP CC": "co2 stored",
        "urban central solid biomass CHP CC": "co2 stored",
        "process emissions": "co2",
    }

    # Filter
    carr = [carrier]
    bus_carr = bus_carrier_selected.get(carr[0], carr)
    unit = "TWh"
    co2_carriers = ["co2", "co2 stored", "process emissions"]
    if bus_carr in co2_carriers:
        unit = "Mt"
    data = balances.query("(carrier in @carr) & (bus_carrier==@bus_carr)").copy()
    data = data.groupby(["planning_horizon", "component", "name", "lt_run", "run_type"], observed=True).sum().reset_index()
    
    data.loc[:, "value"] = data["value"].div(1e6).round(3).abs() # t to Mt

    # Fix order of lt_runs
    data["lt_run"] = pd.Categorical(data["lt_run"], categories=lt_order_nice_names, ordered=True)

    vmin = data["value"].min()
    vmax = data["value"].max()

    plt.rc("font", **plotting["font"])

    # Scatter plot 3x3
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize,
    )

    for i, year in enumerate(planning_horizons):
        year = str(year)
        ax = axes[i]

        subset = data.query("planning_horizon == @year").copy()
        
        # Add custom hue column
    
        sns.scatterplot(
            data=subset.query("run_type == 'LT'"),
            x="lt_run",
            y="value",
            alpha=0.7,
            style="name",
            hue="name",
            palette=custom_hues,
            markers=custom_markers,
            ax=ax,
        )

        sns.scatterplot(
            data=subset.query("run_type != 'LT'"),
            x="lt_run",
            y="value",
            alpha=0.85,
            style="name",
            hue="name",
            palette=custom_hues,
            markers=custom_markers,
            ax=ax,
        )
        
        # turn off legend
        ax.legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(year, fontsize=fontsize)
        
        # Change font size of xticks
        ax.tick_params(axis="x", labelsize=subfontsize)
        ax.tick_params(axis="y", labelsize=subfontsize)

        if i != 0:
            ax.tick_params(left=False, labelleft=False)

        # Vertical line: Connect only the min and max value for each lt_run
        for lt_run, group in subset.groupby("lt_run", observed=True):
            # Get min and max rows
            min_row = group.loc[group["value"].idxmin()]
            max_row = group.loc[group["value"].idxmax()]
            
            x = [lt_run, lt_run]
            y = [min_row["value"], max_row["value"]]
            
            ax.plot(x, y, color="grey", alpha=1, linewidth=0.5, linestyle=":")

        # # Add vertical lines between lt_runs
        # for i, lt_run in enumerate(subset["lt_run"].unique()[:-1]):
        #     ax.axvline(x=i+.5, color="grey", alpha=0.5, linestyle="--", linewidth=0.5)

        # Fill area: spanned by min and max values of each lt_run
        ax.fill_between(
            x=lt_order_nice_names,
            y1=subset.set_index("lt_run").groupby("lt_run", observed=True).min().reindex(lt_order_nice_names)["value"],
            y2=subset.set_index("lt_run").groupby("lt_run", observed=True).max().reindex(lt_order_nice_names)["value"],
            color="grey",
            alpha=0.1,
        )

        # Horizontal line: Add dashed lines between points of same lt_run
        for i, group in subset.groupby("name", observed=True):
            group = group.set_index("lt_run").reindex(lt_order_nice_names).reset_index()

            x = group["lt_run"].values
            y = group["value"].values
            color = custom_hues.get(group.name.iloc[0], "grey")  # fallback to grey if name is missing
            ax.plot(x, y, color=color, alpha=1, linestyle=":", linewidth=0.5)

        # Set ylim
        ax.set_ylim(vmin*0.95, vmax*1.05)

    # Set y label only in first figure
    axes[1].set_xlabel("Long-term run", fontsize=fontsize)
    axes[0].set_ylabel(f"{bus_carrier_nice_name.get(bus_carr, bus_carr)} ({unit} p.a.)", fontsize=fontsize)

    # Add legend below figure once
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
        fontsize=subfontsize,
        frameon=False,
    )
    fig.tight_layout()
    
    fig.savefig(
        snakemake.output[0],
        dpi=dpi,
        bbox_inches="tight",
    )