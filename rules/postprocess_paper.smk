# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule create_paper_plots:
    input:
        regret_matrix=EXPORT_PATH + "/regret_matrix.pdf",
        pcipmi_map=expand(
            EXPORT_PATH + "/map_adm_pcipmi.pdf",
        ),
        totex_heatmap=EXPORT_PATH + "/totex_heatmap.pdf",
        delta_balances=expand(
           EXPORT_PATH + "/delta_balances_{carrier}.pdf",
            **config["scenario"],
            carrier=config_provider("plotting", "figures", "plot_delta_balances", "carriers"),
        ),
        costs_overview=EXPORT_PATH + "/costs_overview.pdf",
        costs_overview_extend=EXPORT_PATH + "/costs_overview_extended.pdf",
        balances_overview=expand(
            EXPORT_PATH + "/balances_overview_{carrier}.pdf",
            **config["scenario"],
            carrier=config_provider("plotting", "figures", "plot_balances_overview", "carriers"),
        ),
        balances_overview_ext=expand(
            EXPORT_PATH + "/balances_overview_extended_{carrier}.pdf",
            **config["scenario"],
            carrier=config_provider("plotting", "figures", "plot_balances_overview", "carriers"),
        ),
        capacities_overview=EXPORT_PATH + "/capacities_overview.pdf",
        capacities_overview_ext= EXPORT_PATH + "/capacities_overview_extended.pdf",
        exogenous_demand= EXPORT_PATH + "/exogenous_demand.pdf",


rule plot_pcipmi_map:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_pcipmi_map"),
    input:
        regions_onshore = resources("regions_onshore_base_s_{clusters}.geojson"),
        regions_offshore = resources("regions_offshore_base_s_{clusters}.geojson"),
        sequestration_potential=resources(
            "co2_sequestration_potential_base_s_{clusters}.geojson"
        ),
        links_co2_pipeline = "data/pcipmi_projects/links_co2_pipeline.geojson",
        links_h2_pipeline = "data/pcipmi_projects/links_h2_pipeline.geojson",
        stores_co2 = "data/pcipmi_projects/stores_co2.geojson",
        stores_h2 = "data/pcipmi_projects/stores_h2.geojson",
    output:
        map=EXPORT_PATH + "/map_{clusters}_{run}.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_pcipmi_map_{clusters}_{run}.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_pcipmi_map_{clusters}_{run}",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_pcipmi_map.py"

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
            + "csvs/{column}/individual/nodal_costs_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        nodal_capacities=expand(
            RESULTS
            + "csvs/{column}/individual/nodal_capacities_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        nodal_capacity_factors=expand(
            RESULTS
            + "csvs/{column}/individual/nodal_capacity_factors_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        capacity_factors=expand(
            RESULTS
            + "csvs/{column}/individual/capacity_factors_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        costs=expand(
            RESULTS
            + "csvs/{column}/individual/costs_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        capacities=expand(
            RESULTS
            + "csvs/{column}/individual/capacities_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        curtailment=expand(
            RESULTS
            + "csvs/{column}/individual/curtailment_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        energy=expand(
            RESULTS
            + "csvs/{column}/individual/energy_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        energy_balance=expand(
            RESULTS
            + "csvs/{column}/individual/energy_balance_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        nodal_energy_balance=expand(
            RESULTS
            + "csvs/{column}/individual/nodal_energy_balance_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        prices=expand(
            RESULTS
            + "csvs/{column}/individual/prices_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        weighted_prices=expand(
            RESULTS
            + "csvs/{column}/individual/weighted_prices_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        market_values=expand(
            RESULTS
            + "csvs/{column}/individual/market_values_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
        metrics=expand(
            RESULTS
            + "csvs/{column}/individual/metrics_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            **config["scenario"],
            allow_missing=True,
        ),
    output:
        costs=RESULTS + "csvs/{column}/costs.csv",
        capacities=RESULTS + "csvs/{column}/capacities.csv",
        energy=RESULTS + "csvs/{column}/energy.csv",
        energy_balance=RESULTS + "csvs/{column}/energy_balance.csv",
        capacity_factors=RESULTS + "csvs/{column}/capacity_factors.csv",
        metrics=RESULTS + "csvs/{column}/metrics.csv",
        curtailment=RESULTS + "csvs/{column}/curtailment.csv",
        prices=RESULTS + "csvs/{column}/prices.csv",
        weighted_prices=RESULTS + "csvs/{column}/weighted_prices.csv",
        market_values=RESULTS + "csvs/{column}/market_values.csv",
        nodal_costs=RESULTS + "csvs/{column}/nodal_costs.csv",
        nodal_capacities=RESULTS + "csvs/{column}/nodal_capacities.csv",
        nodal_energy_balance=RESULTS + "csvs/{column}/nodal_energy_balance.csv",
        nodal_capacity_factors=RESULTS + "csvs/{column}/nodal_capacity_factors.csv",
    threads: 1
    resources:
        mem_mb=8000,
    log:
        RESULTS + "logs/make_global_summary_column_{column}.log",
    benchmark:
        RESULTS + "benchmarks/make_global_summary_{column}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_global_summary.py"


rule make_global_summary_columns:
    input:
        costs=expand(
            RESULTS + "csvs/{column}/costs.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        capacities=expand(
            RESULTS + "csvs/{column}/capacities.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        energy=expand(
            RESULTS + "csvs/{column}/energy.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        energy_balance=expand(
            RESULTS + "csvs/{column}/energy_balance.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        capacity_factors=expand(
            RESULTS + "csvs/{column}/capacity_factors.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        metrics=expand(
            RESULTS + "csvs/{column}/metrics.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        curtailment=expand(
            RESULTS + "csvs/{column}/curtailment.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        prices=expand(
            RESULTS + "csvs/{column}/prices.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        weighted_prices=expand(
            RESULTS + "csvs/{column}/weighted_prices.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        market_values=expand(
            RESULTS + "csvs/{column}/market_values.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        nodal_costs=expand(
            RESULTS + "csvs/{column}/nodal_costs.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        nodal_capacities=expand(
            RESULTS + "csvs/{column}/nodal_capacities.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        nodal_energy_balance=expand(
            RESULTS + "csvs/{column}/nodal_energy_balance.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        nodal_capacity_factors=expand(
            RESULTS + "csvs/{column}/nodal_capacity_factors.csv",
            **config["scenario"],
            allow_missing=True,
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),


rule plot_summary_column:
    params:
        countries=config_provider("countries"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        emissions_scope=config_provider("energy", "emissions"),
        plotting=config_provider("plotting"),
        foresight=config_provider("foresight"),
        co2_budget=config_provider("co2_budget"),
        sector=config_provider("sector"),
        RDIR=RDIR,
    input:
        costs=RESULTS + "csvs/{column}/costs.csv",
        energy=RESULTS + "csvs/{column}/energy.csv",
        balances=RESULTS + "csvs/{column}/energy_balance.csv",
        eurostat="data/eurostat/Balances-April2023",
        co2="data/bundle/eea/UNFCCC_v23.csv",
    output:
        costs=RESULTS + "graphs/{column}/costs.svg",
        energy=RESULTS + "graphs/{column}/energy.svg",
        balances=RESULTS + "graphs/{column}/balances-energy.svg",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/plot_summary_{column}.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_summary.py"


rule plot_summary_columns:
    input:
        costs=expand(
            RESULTS + "graphs/{column}/costs.svg",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        energy=expand(
            RESULTS + "graphs/{column}/energy.svg",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
        balances=expand(
            RESULTS + "graphs/{column}/balances-energy.svg",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),


rule plot_balance_map_column:
    params:
        plotting=config_provider("plotting"),
        plotting_all=config_provider("plotting", "all"),
    input:
        network=RESULTS
        + "networks/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        regions=resources("regions_onshore_base_s_{clusters}.geojson"),
    output:
        RESULTS
        + "maps/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}-balance_map_{carrier}.jpg",
    threads: 1
    resources:
        mem_mb=8000,
    log:
        RESULTS
        + "logs/plot_balance_map/{column}base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{carrier}.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/plot_balance_map/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}_{carrier}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_balance_map.py"


rule plot_balance_map_columns:
    input:
        lambda w: expand(
                RESULTS
                + "maps/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}-balance_map_{carrier}.jpg",
                **config["scenario"],
                run=config["run"]["name"],
                column=config["solve_operations"]["columns"],
                carrier=config_provider("plotting", "balance_map", "bus_carriers")(w),
            ),


rule plot_totex_heatmap:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_totex_heatmap"),
    input:
        longterm=expand(
            RESULTS + "csvs/costs.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    output:
        plot=EXPORT_PATH + "/totex_heatmap.pdf",
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
        longterm=expand(
            RESULTS + "csvs/costs.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
        shortterm=expand(
            RESULTS + "csvs/{column}/costs.csv",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
    output:
        plot=EXPORT_PATH + "/regret_matrix.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_regret_matrix.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_regret_matrix",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_regret_matrix.py"


rule plot_installed_capacities:
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
        plot="results/" + PREFIX + "/plots/installed_capacities_{run}.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_delta_system_costs_{run}.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_delta_system_costs_{run}",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_installed_capacities.py"


rule plot_delta_balances:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_delta_balances"),
    input:
        longterm=expand(
            RESULTS + "csvs/energy_balance.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
        shortterm=expand(
            RESULTS + "csvs/{column}/energy_balance.csv",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
    output:
        # plot="results/" + PREFIX + "/plots/delta_balances_{carrier}.pdf",
        plot= EXPORT_PATH + "/delta_balances_{carrier}.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_delta_balances_{carrier}.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_delta_balances_{carrier}",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_delta_balances.py"


rule plot_balances_overview:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_balances_overview"),
    input:
        longterm=expand(
            RESULTS + "csvs/energy_balance.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
        shortterm=expand(
            RESULTS + "csvs/{column}/energy_balance.csv",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
    output:
        plot= EXPORT_PATH + "/balances_overview_{carrier}.pdf",
        plot_extended= EXPORT_PATH + "/balances_overview_extended_{carrier}.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_balances_overview_{carrier}.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_balances_overview_{carrier}",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_balances_overview.py"


rule plot_costs_overview:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_costs_overview"),
    input:
        longterm=expand(
            RESULTS + "csvs/costs.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
        shortterm=expand(
            RESULTS + "csvs/{column}/costs.csv",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
    output:
        plot= EXPORT_PATH + "/costs_overview.pdf",
        plot_extended= EXPORT_PATH + "/costs_overview_extended.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_costs_overview.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_costs_overview",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_costs_overview.py"


rule plot_capacities_overview:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_capacities_overview"),
    input:
        longterm=expand(
            RESULTS + "csvs/capacities.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
        shortterm=expand(
            RESULTS + "csvs/{column}/capacities.csv",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
    output:
        plot= EXPORT_PATH + "/capacities_overview.pdf",
        plot_extended= EXPORT_PATH + "/capacities_overview_extended.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_capacities_overview.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_capacities_overview",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_capacities_overview.py"


rule plot_exogenous_demand:
    params:
        plotting_all=config_provider("plotting", "all"),
        plotting_fig=config_provider("plotting", "figures", "plot_exogenous_demand"),
    input:
        longterm=expand(
            RESULTS + "csvs/energy_balance.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    output:
        plot= EXPORT_PATH + "/exogenous_demand.pdf",
    log:
        "results/" + PREFIX + "/logs/plot_exogenous_demand.log",
    benchmark:
        "results/" + PREFIX + "/benchmark/plot_exogenous_demand",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_exogenous_demand.py"