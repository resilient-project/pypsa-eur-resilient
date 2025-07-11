# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot costs for all scenarios side-by-side for certain carrier.
"""

import logging
import ast
import matplotlib.pyplot as plt
import pandas as pd

from _helpers import configure_logging, set_scenario_config
from _tools import update_dict

logger = logging.getLogger(__name__)


def import_csvs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Import costs from long-term and short-term runs.
    """

    data_col = "cost"

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
            "plot_costs_overview",
            configfiles=["config/run5.config.yaml"],
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
    subfontsize = fontsize
    dpi = plotting["dpi"]

    opts = config["scenario"]["opts"][0]
    sector_opts = config["scenario"]["sector_opts"][0]
    font = plotting["font"]
    legend_order = plotting["legend_order"]

    # Drop load shedding if in legend_order
    if "Load shedding" in legend_order:
        legend_order.remove("Load shedding")

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    st_order = [col for col in plotting["short_term_run_order"]]
    lt_order = [col for col in plotting["run_order"]]
    lt_order_nice_names = [
        plotting["nice_names"][col] for col in plotting["run_order"]
    ]

    carrier_groups = config["grouping"]
    group_colors = config["group_colors"]

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

    lt_costs = import_csvs(longterm).fillna(0)
    lt_costs["run_type"] = "LT"
    st_costs = import_csvs(shortterm).fillna(0)
    st_costs["run_type"] = "ST"

    costs = pd.concat([lt_costs, st_costs], axis=0).reset_index(drop=True)
    costs["group"] = costs["carrier"].map(carrier_groups)
    costs["group_color"] = costs["group"].map(group_colors)
    
    # to_drop = costs.index[(costs.value.abs()<10)] # Drop small values
    # costs = costs.drop(to_drop, axis=0)
    # costs.reset_index(drop=True, inplace=True)

    # Group by group
    costs = costs.groupby(["planning_horizon", "lt_run", "group", "name", "run_type", "group_color"], observed=True).agg(
        value=("value", "sum"),
    ).div(1e9) # EUR to bn. EUR p.a.
    costs.reset_index(inplace=True)

    # Nice names for lt_run
    costs["lt_run"] = costs["lt_run"].map(plotting["nice_names"])

    costs = costs.sort_values(by=["planning_horizon", "lt_run", "group"]).reset_index(drop=True)    

    # Move name column values to columns
    costs = costs.pivot(
        index=["planning_horizon", "lt_run", "group", "group_color"],
        columns="name",
        values="value",
    ).reset_index()

    # Drop load shedding after debugging
    if "Load shedding" in costs.group.values:
        costs = costs[costs["group"] != "Load shedding"]

    # First plot
    n_lt_runs = costs["lt_run"].nunique()
    n_st_runs = len(st_order)
    n_planning_horizons = len(planning_horizons)

    ymax = costs.loc[costs["Long-term"]>0].groupby(["planning_horizon", "lt_run"], observed=True)["Long-term"].sum().max()
    ymin = 0

    x_anchor = 0
    ncol = 4
    handlelength = 1
    handleheight = 1.1

    xpad = 0.03
    
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_planning_horizons,
        figsize=figsize,
        dpi=dpi,
        sharey=True, 
        tight_layout=True,
    )
    plt.rc("font", **font)

    for i, planning_horizon in enumerate(planning_horizons):
        ax = axes[i]
        planning_horizon = str(planning_horizon)
        data = costs.query("planning_horizon == @planning_horizon").copy().pivot(
            index="lt_run",
            columns="group",
            values="Long-term",
        )

        data_order = [col for col in legend_order if col in data.columns]
        data = data[data_order]

        data = data.reindex(lt_order_nice_names)

        data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.8,
            color=[group_colors.get(col, "yellow") for col in data.columns],
        )

        # Turn off legend
        ax.legend().remove()

        # Set title and labels
        ax.set_xlabel(f"{planning_horizon}", fontsize=fontsize)
        ax.set_ylabel(f"Total system costs (bn. € p.a.)", fontsize=fontsize)

        # Ylim
        ax.set_ylim(ymin, ymax*1.1)

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

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[c], label=c) 
        for c in legend_order[::-1]
    ]

    # Add the production legend (left side, 2 columns)
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor+xpad, 0.03),  # fixed at 0 (left-aligned)
        ncol=ncol,
        fontsize=subfontsize,
        title="",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend.get_title().set_fontweight('bold')
    legend._legend_box.align = "left"    

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
        ncols=n_planning_horizons,
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

        for i, planning_horizon in enumerate(planning_horizons):
            ax = axes[s, i]
            ax = axes[s, i]
            planning_horizon = str(planning_horizon)
            data = costs.query("planning_horizon == @planning_horizon").copy()

            data["delta"] = data[st_run] - data["Long-term"]
            data["delta"] = data["delta"].fillna(0)

            delta_data = data.pivot(
                index="lt_run",
                columns="group",
                values="delta",
            )

            # Reindex
            data_order = [col for col in legend_order if col in delta_data.columns]
            delta_data = delta_data[data_order]
            delta_data = delta_data.reindex(lt_order_nice_names)

            abs_max = max(
                max(ymax, delta_data[delta_data>0].sum(axis=1).max()), 
                abs(min(ymin, delta_data[delta_data<0].sum(axis=1).min())),
            )
            ymax = plotting.get("delta_ymax", abs_max*1.3)
            ymin = plotting.get("delta_ymin", -abs_max*1.3)

            delta_data.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=0.8,
                color=[group_colors.get(col, "yellow") for col in delta_data.columns],
            )

            # Turn off legend
            ax.legend().remove()

            # Set title and labels
            ax.set_xlabel(f"{planning_horizon}", fontsize=fontsize)
            ax.set_ylabel("")

            # # Ylim
            ax.set_ylim(ymin, ymax)

            ax.set_xticklabels(
                delta_data.index,
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
                ax.set_ylabel(f"$\Delta$Total system costs (bn. € p.a.)", fontsize=fontsize)

            # Add totals of positive values on top
            totals = delta_data[delta_data>0].sum(axis=1)
            net_totals = delta_data.sum(axis=1)
            for j, nt in enumerate(net_totals):
                ax.text(
                    x=j,
                    y=totals.iloc[j],
                    s=f"{nt:+.1f}" if abs(nt) > 0.1 else "",
                    ha="center",
                    va="bottom",
                    fontsize=subfontsize,
                )

            # Add 0 axis line
            ax.axhline(0, color="black", lw=0.5)

        # Add label of st_run in each row
        axes[s, 0].text(
            x=-0.5,
            y=ymax*0.92,
            s=plotting["nice_names"][st_run],
            ha="left",
            va="center",
            fontsize=subfontsize-2,
            rotation=0,
        )

    # Change font size of major sharey ticks
    for ax in axes.flatten():
        ax.tick_params(axis="y", labelsize=subfontsize)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[c], label=c)
        for c in legend_order[::-1]
    ]

    # Add the production legend (left side, 2 columns)
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor+xpad, 0.02),  # fixed at 0 (left-aligned)
        ncol=ncol,
        fontsize=subfontsize,
        title="",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend.get_title().set_fontweight('bold')
    legend._legend_box.align = "left"    
    
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

    