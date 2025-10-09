# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot costs for all scenarios side-by-side for certain carrier.
"""

import logging
import ast
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
            "plot_costs_overview_sensitivity",
            configfiles=["config/postprocess_sensitivities.config.yaml"],
            sensitivity="pipelines1.3",
            )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config
    plotting = config["plotting"]["all"]
    plotting = update_dict(plotting, snakemake.params.plotting_fig)
    nice_names = config["plotting"]["nice_names"]
    tech_colors = config["plotting"]["tech_colors"]
    sensitivity = snakemake.wildcards.sensitivity

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
    runs = pd.DataFrame()
    runs["path"] = snakemake.input.runs
    runs["prefix"] = runs["path"].apply(lambda x: x.split("/")[-4])
    runs["lt_run"] = runs["path"].apply(lambda x: x.split("/")[-3])

    # Split strings by __
    runs["name"] = runs["lt_run"].apply(lambda x: x.split("___")[0])
    runs["lt_run"] = runs["lt_run"].apply(lambda x: x.split("___")[-1])
    
    # Rename main runs
    runs.loc[runs.prefix=="pcipmi", "name"] = "main"

    # Store list of sensitivities
    sensitivities = sorted(runs.query("prefix=='sensitivities'")["name"].unique())

    costs = import_csvs(runs).fillna(0)
    costs["group"] = costs["carrier"].map(carrier_groups)
    costs["group_color"] = costs["group"].map(group_colors)

    # Group by group
    costs = costs.groupby(["planning_horizon", "lt_run", "group", "name", "group_color"], observed=True).agg(
        value=("value", "sum"),
    ).div(1e9) # EUR to bn. EUR p.a.
    costs.reset_index(inplace=True)
    costs["lt_run"] = costs["lt_run"].map(plotting["nice_names"])

    costs = costs.sort_values(by=["planning_horizon", "lt_run", "group"]).reset_index(drop=True)    

    # Move name column values to columns
    costs = costs.pivot(
        index=["planning_horizon", "lt_run", "group", "group_color"],
        columns="name",
        values="value",
    ).reset_index()

    main_totals = costs[["planning_horizon", "lt_run", "main"]].groupby(["planning_horizon", "lt_run"]).sum()

    delta_to_main = costs.copy()
    delta_to_main[sensitivities] = delta_to_main[sensitivities].subtract(delta_to_main["main"], axis=0)

    delta_abs_max = delta_to_main.groupby(["planning_horizon", "lt_run"]).sum()[sensitivity].abs().max()

    # Drop load shedding after debugging
    if "Load shedding" in delta_to_main.group.values:
        delta_to_main = delta_to_main[delta_to_main["group"] != "Load shedding"]

    # First plot
    n_lt_runs = delta_to_main["lt_run"].nunique()
    n_st_runs = len(st_order)
    n_planning_horizons = len(planning_horizons)


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
        data = delta_to_main.query("planning_horizon == @planning_horizon").copy().pivot(
            index="lt_run",
            columns="group",
            values=sensitivity,
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
        ax.set_ylabel(f"$\Delta$Total system costs (bn. € p.a.)", fontsize=fontsize)

        # Ylim
        # ax.set_ylim(-20, 20)

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

        # Net totals (including negatives)
        totals = data.sum(axis=1)

        # Position for label = top of positive stack
        positive_tops = data.clip(lower=0).sum(axis=1)

        for j, total in enumerate(totals):
            if total != 0:
                sign = "+" if total > 0 else "-"
                fmt = ".0f" if abs(total) >= 10 else ".1f"
                y_label = positive_tops[j] if positive_tops[j] > 0 else 0

                rel_value = abs(total) / main_totals.loc[(planning_horizon, data.index[j]), "main"]
                rel_value_str = f"{rel_value:.1%}" #if rel_value >= 0.01 else "<1%"

                # main value
                ax.text(
                    x=j,
                    y=y_label,
                    s=f"{sign}{abs(total):{fmt}}",
                    ha="center",
                    va="bottom",
                    fontsize=subfontsize,
                )

                # smaller relative value just below
                ax.text(
                    x=j,
                    y=total-0.03*delta_abs_max,  # adjust vertical offset if needed
                    s=f"{sign}{rel_value_str}",
                    ha="center",
                    va="top",
                    fontsize=subfontsize * 0.6,
                )

                ax.plot(
                    j, total,
                    marker=".",
                    color="black",
                    markersize=1,
                    zorder=5,
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

    handles.append(
        Line2D(
            [0], [0],
            marker="o",
            color="black",
            linestyle="",
            markersize=1,
            label="Net change",
        )
    )

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
