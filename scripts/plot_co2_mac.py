# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from _helpers import configure_logging, set_scenario_config
logger = logging.getLogger(__name__)


def import_csvs(
    csvs: list,
    **kwargs,
    ) -> pd.DataFrame:
    """
    Import costs from long-term and short-term runs.
    """
    kwargs.setdefault("index_col", None)
    kwargs.setdefault("header", None)
    kwargs.setdefault("axis", 0)
    kwargs.setdefault("skiprows", 0)

    df_list = []
    for csv in csvs:
        df = pd.read_csv(csv, skiprows=kwargs["skiprows"], index_col=kwargs["index_col"], header=kwargs["header"])
        df["run"] = csv.split("/")[-3]
        df_list.append(df)
    
    dfs = pd.concat(df_list, axis=kwargs["axis"])
    return dfs


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_co2_mac",
            configfiles=["config/config.co2-sweep.yaml"],
            )
        
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    total_co2_emissions = 4603917344.131599 # tCO2

    config = snakemake.config
    configs = snakemake.input.configs
    costs = snakemake.input.costs
    metrics = snakemake.input.metrics

    carrier_groups = config["grouping"]
    color_dict = config["group_colors"]
    carrier_group_order = config["group_order"]

    # Individual configs
    configs = dict(
        zip(
            [c.split("/")[-3].split(".")[0] for c in configs],
            [yaml.safe_load(open(c, "r")) for c in configs],
        )

    )

    # Scenario data
    scenario_data = pd.DataFrame()
    scenario_data["run"] = configs.keys()
    scenario_data["co2_budget_mt"] = scenario_data["run"].map(lambda r: configs[r]["co2_budget"][2050]) * total_co2_emissions / 1e6
    scenario_data["co2_seq_potential_mt"] = scenario_data["run"].map(lambda r: configs[r]["sector"]["co2_sequestration_potential"][2050])
    scenario_data["cost_year"] = scenario_data["run"].map(lambda r: configs[r]["costs"]["year"])
    scenario_data.set_index("run", inplace=True)

    # Import results
    costs = import_csvs(costs, skiprows=3, header=0)
    costs.columns = ["cost", "component", "carrier", "value", "run"]
    costs["value"] = costs["value"].div(1e9) # Convert to bn. euros

    # Update relative CO2 budget with absolute vaue
    metrics = import_csvs(metrics).fillna(0)
    metrics.columns = ["metric", "value", "run"]

    # Energy balance
    eb = import_csvs(snakemake.input.energy_balance, skiprows=3, header=0)
    eb.columns = ["component", "carrier", "bus_carrier", "value", "run"]
    
    # Weighted prices
    weighted_prices = import_csvs(snakemake.input.weighted_prices, skiprows=3)
    weighted_prices.columns = ["carrier", "value", "run"]

    # Map aggregate results to scenario data
    scenario_data["system_costs"] = costs.groupby("run").agg(total_value=("value", "sum")).round(1)
    scenario_data["co2_price"] = (metrics.query("metric == 'co2_shadow'").set_index("run")["value"]*(-1)).round(1)
    scenario_data["co2_storage_price"] = (metrics.query("metric == 'co2_storage_shadow'").set_index("run")["value"]).round(1).clip(0)
    scenario_data["co2_sequestered"] = eb.query("(bus_carrier == 'co2 sequestered') & (component == 'Link')").set_index("run")["value"].div(1e6).round(1)  
    scenario_data["ac_price"] = weighted_prices.query("carrier=='AC'").set_index("run")["value"].round(1)
    scenario_data["h2_price"] = weighted_prices.query("carrier=='H2'").set_index("run")["value"].round(1)
    scenario_data["biogas_price"] = weighted_prices.query("carrier=='biogas'").set_index("run")["value"].round(1)
    scenario_data["industry_methanol_price"] = weighted_prices.query("carrier=='industry methanol'").set_index("run")["value"].round(1)
    scenario_data["naphta_for_industry_price"] = weighted_prices.query("carrier=='naphta for industry'").set_index("run")["value"].round(1)
    scenario_data["kerosene_for_aviation_price"] = weighted_prices.query("carrier=='kerosene for aviation'").set_index("run")["value"].round(1)

    # Grouped costs
    costs_grouped = costs.set_index("run")
    costs_grouped["carrier_group"] = (
        costs_grouped["carrier"]
        .map(lambda c: carrier_groups.get(c, "NOGROUP"))
    )
    costs_grouped = costs_grouped.groupby(["run", "carrier_group"]).agg(costs=("value", "sum")).reset_index().set_index("run")
    costs_grouped[["co2_budget_mt", "co2_seq_potential_mt", "cost_year"]] = scenario_data[["co2_budget_mt", "co2_seq_potential_mt", "cost_year"]]

    # Check whether number of elements in group_order and costs_grouped.carrier_group.unique() match
    b_group_ok = set(costs_grouped["carrier_group"].unique()) == set(carrier_group_order)
    if not b_group_ok:
        logger.warning(
            "Number of carrier groups in config/group_order does not match number of carrier groups in costs data"
        )
        logger.warning(f"Groups in costs data: {costs_grouped['carrier_group'].unique()}")
        logger.warning(f"Groups in config: {carrier_group_order}")
    
    # Define values to loop over
    seq_potentials = sorted({configs[i[1]]["sector"]["co2_sequestration_potential"][2050] for i in enumerate(configs)})
    cost_years = sorted({configs[i[1]]["costs"]["year"] for i in enumerate(configs)})

    # Plot
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(len(seq_potentials), len(cost_years), figsize=(12, 20), sharex=True, sharey=True)
    plt.rcParams.update({'font.size': 11})

    ymax = scenario_data["co2_price"].max()
    ymax2 = scenario_data["system_costs"].max()

    for i, seq in enumerate(seq_potentials):
        for j, cy in enumerate(cost_years):
            ax = axes[i, j]

            subset = scenario_data.query(f"(co2_seq_potential_mt == {seq}) & (cost_year == {cy})")
            subset = subset.sort_values(by="co2_budget_mt", ascending=False)

            costs_subset = (
                costs_grouped.query(f"co2_seq_potential_mt == {seq} & cost_year == {cy}")
            ).pivot_table(
                index="co2_budget_mt",
                columns="carrier_group",
                values="costs",
                aggfunc="sum",
            ).clip(lower=0)
            # Reorder columns
            costs_subset = costs_subset.reindex(columns=carrier_group_order)

            # Right y-axis: costs
            ax2 = ax.twinx()
            costs_subset.plot.area(
                ax=ax2,
                stacked=True,
                color=[color_dict.get(c, "#333333") for c in costs_subset.columns],
                alpha=0.8,
                linewidth=0,
                zorder=1,
            )

            # Flip z-orders
            ax.set_zorder(ax2.get_zorder() + 1)  # move primary axis on top
            ax.patch.set_visible(False) 


            # Left y-axis: CO2 price
            sns.lineplot(
                data=subset,
                x="co2_budget_mt",
                y="co2_price",
                marker="o",
                ax=ax,
                # label="CO$_2$ price",
                color='blue',
                zorder=5,
                label="CO$_2$ price",
            )

            # Vertical reference lines
            ax.axvline(x=0.45 * total_co2_emissions / 1e6, color='grey', linestyle='--', linewidth=0.7)
            ax.text(0.44 * total_co2_emissions / 1e6, ymax, "-55% (~today)",  fontsize=10, ha='left', va='top')
            ax.axvline(x=0.65 * total_co2_emissions / 1e6, color='grey', linestyle='--', linewidth=0.7)
            ax.text(0.64 * total_co2_emissions / 1e6, ymax, "-35%", fontsize=10, ha='left', va='top')
            ax.axvline(x=-0.02*total_co2_emissions/1e6, color='grey', linestyle='--', linewidth=0.7)
            ax.text(0, ymax, "-102%", fontsize=10, ha='right', va='top')

            ax.set_title(f"CY {cy} $\\cdot$ CO$_2$ seq. {seq} Mta$^{{-1}}$")

            # Remove individual x and y labels (common labels added later)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax2.set_ylabel("")
            
            ax.set_ylim(0, ymax*1.05)
            ax2.set_ylim(0, ymax2*1.05)
            ax.grid(False)
            ax2.grid(False)

            ax.legend_.remove() if ax.get_legend() else None  # remove subplot legend
            ax2.legend_.remove() if ax2.get_legend() else None  # remove subplot legend

    fig.gca().invert_xaxis()

    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    # Add legend
    # Collect handles/labels from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Merge while preserving order
    handles = handles1 + list(reversed(handles2))
    labels = labels1 + list(reversed(labels2))

    # Add combined legend to figure
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        bbox_transform=fig.transFigure,
        frameon=False,
    )
    
    # Add common axis labels
    fig.text(0.5, -0.01, "CO$_2$ emissions (Mta$^{-1}$)", ha='center')
    fig.text(-0.01, 0.5, "CO$_2$ price (€tCO$_2^{-1}$)", va='center', rotation='vertical')
    fig.text(1.0, 0.5, "Total system costs (bn.€a$^{-1}$)", va='center', rotation='vertical',)

    plt.tight_layout()  # Adjust layout for right y-label
    plt.show()
