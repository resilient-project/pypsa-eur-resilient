# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule create_paper_plots:
    input:
        totex_heatmap="results/" + PREFIX + "/plots/totex_heatmap.pdf",
        delta_system_costs=expand(
            "results/" + PREFIX + "/plots/delta_system_costs_{planning_horizons}.pdf",
            **config["scenario"],
        )


rule make_summary_column:
    input:
        network=RESULTS + "networks/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    output:
        nodal_costs=RESULTS
        + "csvs/{column}/individual/nodal_costs_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        nodal_capacities=RESULTS
        + "csvs/{column}/individual/nodal_capacities_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        nodal_capacity_factors=RESULTS
        + "csvs/{column}/individual/nodal_capacity_factors_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        capacity_factors=RESULTS
        + "csvs/{column}/individual/capacity_factors_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        costs=RESULTS
        + "csvs/{column}/individual/costs_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        capacities=RESULTS
        + "csvs/{column}/individual/capacities_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        curtailment=RESULTS
        + "csvs/{column}/individual/curtailment_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        energy=RESULTS
        + "csvs/{column}/individual/energy_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        energy_balance=RESULTS
        + "csvs/{column}/individual/energy_balance_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        nodal_energy_balance=RESULTS
        + "csvs/{column}/individual/nodal_energy_balance_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        prices=RESULTS
        + "csvs/{column}/individual/prices_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        weighted_prices=RESULTS
        + "csvs/{column}/individual/weighted_prices_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        market_values=RESULTS
        + "csvs/{column}/individual/market_values_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        metrics=RESULTS
        + "csvs/{column}/individual/metrics_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    threads: 1
    resources:
        mem_mb=8000,
    log:
        RESULTS
        + "logs/{column}/make_summary_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/{column}/make_summary_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_summary.py"


rule make_global_summary_column:
    params:
        scenario=config_provider("scenario"),
        RDIR=RDIR,
    input:
        nodal_costs=expand(
            RESULTS
            + "csvs/individual/nodal_costs_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        nodal_capacities=expand(
            RESULTS
            + "csvs/individual/nodal_capacities_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        nodal_capacity_factors=expand(
            RESULTS
            + "csvs/individual/nodal_capacity_factors_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        capacity_factors=expand(
            RESULTS
            + "csvs/individual/capacity_factors_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        costs=expand(
            RESULTS
            + "csvs/individual/costs_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        capacities=expand(
            RESULTS
            + "csvs/individual/capacities_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        curtailment=expand(
            RESULTS
            + "csvs/individual/curtailment_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        energy=expand(
            RESULTS
            + "csvs/individual/energy_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        energy_balance=expand(
            RESULTS
            + "csvs/individual/energy_balance_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        nodal_energy_balance=expand(
            RESULTS
            + "csvs/individual/nodal_energy_balance_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        prices=expand(
            RESULTS
            + "csvs/individual/prices_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        weighted_prices=expand(
            RESULTS
            + "csvs/individual/weighted_prices_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        market_values=expand(
            RESULTS
            + "csvs/individual/market_values_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        metrics=expand(
            RESULTS
            + "csvs/individual/metrics_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
    output:
        costs=RESULTS + "csvs/costs.csv",
        capacities=RESULTS + "csvs/capacities.csv",
        energy=RESULTS + "csvs/energy.csv",
        energy_balance=RESULTS + "csvs/energy_balance.csv",
        capacity_factors=RESULTS + "csvs/capacity_factors.csv",
        metrics=RESULTS + "csvs/metrics.csv",
        curtailment=RESULTS + "csvs/curtailment.csv",
        prices=RESULTS + "csvs/prices.csv",
        weighted_prices=RESULTS + "csvs/weighted_prices.csv",
        market_values=RESULTS + "csvs/market_values.csv",
        nodal_costs=RESULTS + "csvs/nodal_costs.csv",
        nodal_capacities=RESULTS + "csvs/nodal_capacities.csv",
        nodal_energy_balance=RESULTS + "csvs/nodal_energy_balance.csv",
        nodal_capacity_factors=RESULTS + "csvs/nodal_capacity_factors.csv",
    threads: 1
    resources:
        mem_mb=8000,
    log:
        RESULTS + "logs/make_global_summary.log",
    benchmark:
        RESULTS + "benchmarks/make_global_summary"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_global_summary.py"


rule plot_totex_heatmap:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_totex_heatmap"),
    input:
        expand(
            RESULTS 
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            run=config["run"]["name"],
            allow_missing=True
        ),
    output:
        plot="results/" + PREFIX + "/plots/totex_heatmap.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_totex_heatmap.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_totex_heatmap",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_totex_heatmap.py"


rule plot_delta_system_costs:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_delta_system_costs"),
    input:
        expand(
            RESULTS 
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            run=config["run"]["name"],
            allow_missing=True
        ),
    output:
        plot="results/" + PREFIX + "/plots/delta_system_costs_{planning_horizons}.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_delta_system_costs_{planning_horizons}.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_delta_system_costs_{planning_horizons}",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_delta_system_costs.py"


rule plot_regret_matrix:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_regret_matrix"),
    input:
        rows=expand(
            RESULTS 
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            run=config["run"]["name"],
            allow_missing=True
        ),
        columns=expand(
            RESULTS 
            + "networks/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
    output:
        plot="results/" + PREFIX + "/plots/regret_matrix.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_regret_matrix.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_regret_matrix",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_regret_heatmap.py"