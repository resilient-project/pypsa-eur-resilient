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