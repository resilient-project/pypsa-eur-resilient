# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule solve_operations_sector_network:
    params:
        options=config_provider("solving", "options"),
        solving=config_provider("solving"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        solve_operations=config_provider("solve_operations"),
    input:
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    output:
        network=RESULTS + "networks/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    log:
        solver=RESULTS
        + "logs/operations/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        memory=RESULTS
        + "logs/operations/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
        python=RESULTS
        + "logs/operations/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_operations_sector_network/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    threads: 4
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_operations_sector_network.py"


rule solve_operations_sector_networks:
    input:
        expand(
            RESULTS 
            + "networks/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),
