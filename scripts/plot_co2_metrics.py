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

def split_str_to_cols(
    df: pd.DataFrame,
    col: str,
    col_rename: list | None = None,
    sep: str = ",",
    ) -> pd.DataFrame:
    """
    Split a string column into multiple columns.
    """
    # Split the column and expand into new columns
    split_cols = df[col].str.split(sep, expand=True)
    
    if col_rename:
        if len(col_rename) != split_cols.shape[1]:
            raise ValueError(
                f"Number of new column names ({len(col_rename)}) does not match number of split columns ({split_cols.shape[1]})"
            )
        split_cols.columns = col_rename
    
    # Concatenate the original DataFrame with the new columns
    df = pd.concat([df, split_cols], axis=1)

    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_co2_metrics",
            configfiles=["config/config.co2-sweep.yaml"],
            )
        
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    total_co2_emissions = 4603917344.131599 # tCO2

    config = snakemake.config
    configs = snakemake.input.configs
    costs = snakemake.input.costs
    metrics = snakemake.input.metrics

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

    # Update relative CO2 budget with absolute vaue
    metrics = import_csvs(metrics).fillna(0)
    metrics.columns = ["metric", "value", "run"]

    # Energy balance
    eb = import_csvs(snakemake.input.energy_balance, skiprows=3, header=0)
    eb.columns = ["component", "carrier", "bus_carrier", "value", "run"]

    # Map aggregate results to scenario data
    scenario_data["system_costs"] = costs.groupby("run").agg(total_value=("value", "sum")).round(0)
    scenario_data["co2_price"] = metrics.query("metric == 'co2_shadow'").set_index("run")["value"]*(-1)
    scenario_data["co2_sequestered"] = eb.query("(bus_carrier == 'co2 sequestered') & (component == 'Link')").set_index("run")["value"].div(1e6)  # Convert to Mt

    # Filter subset
    # Ensure nice plot style
    sns.set(style="whitegrid")

    # Define values to loop over
    seq_potentials = [250, 5000]
    years = [2030, 2050]


    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i, seq in enumerate(seq_potentials):
        for j, year in enumerate(years):
            ax = axes[i, j]

            subset = scenario_data.query(f"(co2_seq_potential_mt == {seq}) & (cost_year == {year})")
            subset = subset.sort_values(by="co2_budget_mt", ascending=False)

            subset["co2_abated"] = subset["co2_budget_mt"].diff(+1).fillna(0) * (-1)
            subset["co2_reduction_rel"] = (subset["co2_budget_mt"] / total_co2_emissions * 1e6).round(2)
            subset["co2_abated_cum"] = (1 - subset["co2_reduction_rel"]) * total_co2_emissions / 1e6

            # Vertical reference lines
            ax.axvline(x=0.35 * total_co2_emissions / 1e6, color='red', linestyle='--', linewidth=0.7)
            ax.text(0.35 * total_co2_emissions / 1e6, 350, "-35% (2025)", color='red', fontsize=10, ha='left', va='center')
            ax.axvline(x=0.55 * total_co2_emissions / 1e6, color='red', linestyle='--', linewidth=0.7)
            ax.text(0.55 * total_co2_emissions / 1e6, 350, "-55%", color='red', fontsize=10, ha='left', va='center')
            ax.axvline(x=1 * total_co2_emissions / 1e6, color='red', linestyle='--', linewidth=0.7)
            ax.text(1 * total_co2_emissions / 1e6, 350, "-100%", color='red', fontsize=10, ha='left', va='center')

            # Right y-axis: CO2 sequestered
            ax2 = ax.twinx()
            ax2.fill_between(
                subset["co2_abated_cum"],
                subset["co2_sequestered"],
                color="lightgrey",
                alpha=0.6,
                label="CO₂ sequestered"
            )
            ax2.plot(
                subset["co2_abated_cum"],
                subset["co2_sequestered"],
                marker="x",
                color="green",
                label="CO₂ sequestered"
            )

            # Left y-axis: CO2 price
            sns.lineplot(
                data=subset,
                x="co2_abated_cum",
                y="co2_price",
                marker="o",
                ax=ax,
                # label="CO₂ price",
                color='blue',
                zorder=5,
            )

            ax.set_title(f"Sequestration potential: {seq} Mt, Cost year: {year}")

            # Remove individual x and y labels (common labels added later)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax2.set_ylabel("")
            
            ax2.set_ylim(-50, 1200)
            ax.grid(True)
    
    # Add common axis labels
    fig.text(0.5, 0.08, "Saved CO$_2$ emissions (Mt)", ha='center', fontsize=14)
    fig.text(0.08, 0.5, "CO$_2$ price (€/tCO₂)", va='center', rotation='vertical', fontsize=14)
    fig.text(0.95, 0.5, "CO$_2$ sequestered (Mt)", va='center', rotation='vertical', fontsize=14, color='green')

    plt.tight_layout(rect=[0.1, 0.1, 0.94, 1])  # Adjust layout for right y-label
    plt.show()




