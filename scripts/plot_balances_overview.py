# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot balances for all scenarios side-by-side for certain carrier.
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
            "plot_balances_overview",
            configfiles=["config/run5.config.yaml"],
            carrier="co2 stored",
            )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config
    plotting = config["plotting"]["all"]
    plotting = update_dict(plotting, snakemake.params.plotting_fig)
    nice_names = config["plotting"]["nice_names"]
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

    carrier_nice_name = plotting["carrier_nice_names"][carrier]
    carrier_unit = plotting["carrier_units"][carrier]

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    st_order = [col for col in plotting["short_term_run_order"]]
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

    shortterm = pd.DataFrame()
    shortterm["path"] = snakemake.input.shortterm
    shortterm["name"] = shortterm["path"].apply(lambda x: x.split("/")[-2])
    shortterm["lt_run"] = shortterm["path"].apply(lambda x: x.split("/")[-4])

    lt_balances = import_csvs(longterm).fillna(0)
    lt_balances["run_type"] = "LT"
    st_balances = import_csvs(shortterm).fillna(0)
    st_balances["run_type"] = "ST"

    balances = pd.concat([lt_balances, st_balances], axis=0).reset_index(drop=True)
    to_drop = balances.index[(balances.value.abs()<10)] # Drop small values
    balances = balances.drop(to_drop, axis=0)
    balances.reset_index(drop=True, inplace=True)

    # Drop st
    # balances.drop(
    #     balances.query("run_type=='ST'").index,
    #     axis=0,
    #     inplace=True,
    # )

    # Only keep bus_carrier
    balances = balances.query("bus_carrier==@carrier")
    balances = balances.groupby(["planning_horizon", "lt_run", "carrier", "name", "run_type"], observed=True).agg(
        value=("value", "sum"),
    ).div(1e6) # t to Mt, MWh to TWh
    balances.reset_index(inplace=True)

    balances["nice_name"] = balances["carrier"].replace(nice_names)
    balances = balances.sort_values(by=["planning_horizon", "lt_run", "nice_name"]).reset_index(drop=True)
    

    # Move name column values to columns
    balances = balances.pivot(
        index=["planning_horizon", "lt_run", "carrier"],
        columns="name",
        values="value",
    ).reset_index()
    
    carrier_order = list(balances.carrier.unique())

    # supply carrier
    prod_carriers = sorted(set(balances.loc[balances["Long-term"] > 0, "carrier"]))
    cons_carriers = sorted(set(balances.carrier.unique()) - set(prod_carriers))

    # Drop 
    if "H2 pipeline" in cons_carriers:
        cons_carriers.remove("H2 pipeline")

    # Sort by nice_names
    prod_carriers = sorted(prod_carriers, key=lambda x: nice_names.get(x, x))
    cons_carriers = sorted(cons_carriers, key=lambda x: nice_names.get(x, x))

    # Using seaborn plot create stacked bar plot
    n_lt_runs = balances["lt_run"].nunique()

    ymax = balances.loc[balances["Long-term"]>0].groupby(["planning_horizon", "lt_run"], observed=True)["Long-term"].sum().max()
    ymin = -ymax 

    x_anchor_prod = 0
    x_anchor_cons = 0.67
    ncol_prod = 2
    ncol_cons = 1
    handlelength = 1
    handleheight = 1.1

    if carrier == "H2":
        x_anchor_cons = 0.5
        ncol_cons = 2

    xpad = 0.03
    
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_lt_runs,
        figsize=figsize,
        dpi=dpi,
        sharey=True, 
        tight_layout=True,
    )
    plt.rc("font", **font)

    for i, lt_run in enumerate(lt_order):
        ax = axes[i]
        data = balances.query("lt_run == @lt_run").copy().pivot(
            index="planning_horizon",
            columns="carrier",
            values="Long-term",
        )

        data_order = data.columns.tolist()
        data_order = [col for col in carrier_order if col in data_order]
        data = data[data_order]


        data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.8,
            color=[tech_colors.get(col, "yellow") for col in data.columns],
        )

        # Turn off legend
        ax.legend().remove()

        # Set title and labels
        ax.set_xlabel(f"{plotting["nice_names"][lt_run]}", fontsize=fontsize)
        ax.set_ylabel(f"{carrier_nice_name} balance ({carrier_unit} p.a.)", fontsize=fontsize)

        # Ylim
        ax.set_ylim(ymin*1.3, ymax*1.3)

        ax.set_xticklabels(
            data.index,
            rotation=90,
            fontsize=subfontsize,
        )
        
        # Remove all grid lines
        ax.grid(False)

        # Remove y ticks in all but the first plot
        if i > 0:
            ax.yaxis.set_visible(False)

        # Add totals of positive values on top
        totals = data[data>0].sum(axis=1)
        for j, total in enumerate(totals):
            if total > 0:
                ax.text(
                    x=j,
                    y=total,
                    s=f"{total:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=subfontsize,
                )

        # Add 0 axis line
        ax.axhline(0, color="black", lw=0.5)
        

    # Change font size of major sharey ticks
    for ax in axes:
        ax.tick_params(axis="y", labelsize=subfontsize)

    # Add legend
    # Create legend handles for production and consumption carriers
    prod_handles = [
        plt.Rectangle((0, 0), 1, 1, color=tech_colors[c], label=nice_names.get(c, c))
        for c in prod_carriers
    ]

    cons_handles = [
        plt.Rectangle((0, 0), 1, 1, color=tech_colors[c], label=nice_names.get(c, c))
        for c in cons_carriers
    ]

    # Add the production legend (left side, 2 columns)
    legend_prod = fig.legend(
        handles=prod_handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor_prod+xpad, 0.07),  # fixed at 0 (left-aligned)
        ncol=ncol_prod,
        fontsize=subfontsize,
        title="Production",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend_prod.get_title().set_fontweight('bold')
    legend_prod._legend_box.align = "left"    

    # Add the consumption legend (middle, 1 column)
    legend_cons = fig.legend(
        handles=cons_handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor_cons+xpad, 0.07),  # fixed at 0.5 (centered)
        ncol=ncol_cons,
        fontsize=subfontsize,
        title="Consumption",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend_cons.get_title().set_fontweight('bold')
    legend_cons._legend_box.align = "left"

    # All borders to 0.5 thickness
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("black")

    # Tight layout
    plt.tight_layout()
    
    fig.subplots_adjust(wspace=0.05) 
 
    fig.savefig(
        snakemake.output[0],
        dpi=dpi,
        bbox_inches="tight",
    )

    # Extended plot
    n_st_runs = len(st_order)
    fig, axes = plt.subplots(
        nrows=n_st_runs,
        ncols=n_lt_runs,
        figsize=(figsize[0], figsize[1]*n_st_runs/2),
        dpi=dpi,
        sharey="row", 
        sharex="col",
        tight_layout=True,
    )

    plt.rc("font", **font)

    for s, st_run in enumerate(st_order):
        ymax = 0
        ymin = 0

        for i, lt_run in enumerate(lt_order):
            ax = axes[s, i]
            data = balances.query("lt_run == @lt_run").copy().pivot(
                index="planning_horizon",
                columns="carrier",
                values="Long-term",
            )

            data_st = balances.query("lt_run == @lt_run").copy().pivot(
                index="planning_horizon",
                columns="carrier",
                values=st_run,
            )

            # Reindex
            data_st = data_st.reindex(data.index)
            # Column order
            data_st = data_st.reindex(columns=data.columns)

            delta_data = data_st - data
            delta_data.fillna(0, inplace=True)

            data_order = data.columns.tolist()
            data_order = [col for col in carrier_order if col in data_order]
            delta_data = delta_data[data_order]

            ymax = max(ymax, delta_data[delta_data>0].sum(axis=1).max())
            ymin = -ymax

            delta_data.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=0.8,
                color=[tech_colors.get(col, "yellow") for col in data.columns],
            )

            # Turn off legend
            ax.legend().remove()

            # Set title and labels
            ax.set_xlabel(f"{plotting["nice_names"][lt_run]}", fontsize=fontsize)
            ax.set_ylabel("")

            # # Ylim
            ax.set_ylim(ymin*1.3, ymax*1.3)

            ax.set_xticklabels(
                data.index,
                rotation=90,
                fontsize=subfontsize,
            )
            
            # Remove all grid lines
            ax.grid(False)

            # Remove y ticks in all but the first plot
            if i > 0:
                ax.yaxis.set_visible(False)

            if s < n_st_runs - 1:
                ax.xaxis.set_visible(False)

            if s == 1:
                ax.set_ylabel(f"$\Delta${carrier_nice_name} balance ({carrier_unit} p.a.)", fontsize=fontsize)

            # Add totals of positive values on top
            totals = delta_data[delta_data>0].sum(axis=1)
            for j, total in enumerate(totals):
                if total > 0.1:
                    ax.text(
                        x=j,
                        y=total,
                        s=f"{total:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=subfontsize,
                    )

            # Add 0 axis line
            ax.axhline(0, color="black", lw=0.5)

        # Add label of st_run in each row
        axes[s, 0].text(
            x=1,
            y=ymin*1.15,
            s=plotting["nice_names"][st_run],
            ha="center",
            va="center",
            fontsize=subfontsize-2,
            rotation=0,
        )

    # Change font size of major sharey ticks
    for ax in axes.flatten():
        ax.tick_params(axis="y", labelsize=subfontsize)

    # Add legend
    # Create legend handles for production and consumption carriers
    prod_handles = [
        plt.Rectangle((0, 0), 1, 1, color=tech_colors[c], label=nice_names.get(c, c))
        for c in prod_carriers
    ]

    cons_handles = [
        plt.Rectangle((0, 0), 1, 1, color=tech_colors[c], label=nice_names.get(c, c))
        for c in cons_carriers
    ]

    # Add the production legend (left side, 2 columns)
    legend_prod = fig.legend(
        handles=prod_handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor_prod+xpad, 0.045),  # fixed at 0 (left-aligned)
        ncol=ncol_prod,
        fontsize=subfontsize,
        title="Production",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend_prod.get_title().set_fontweight('bold')
    legend_prod._legend_box.align = "left"    

    # Add the consumption legend (middle, 1 column)
    legend_cons = fig.legend(
        handles=cons_handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor_cons+xpad, 0.045),  # fixed at 0.5 (centered)
        ncol=ncol_cons,
        fontsize=subfontsize,
        title="Consumption",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend_cons.get_title().set_fontweight('bold')
    legend_cons._legend_box.align = "left"

    # All borders to 0.5 thickness
    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("black")

    # Tight layout
    plt.tight_layout()
    
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
 
    fig.savefig(
        snakemake.output[1],
        dpi=dpi,
        bbox_inches="tight",
    )

    