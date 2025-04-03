# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch using the capacities of previous capacity expansion in rule :mod:`solve_network`.
Custom constraints and extra_functionality can be set in the config.
"""

import logging

import numpy as np
import pypsa
from _benchmark import memory_logger
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from solve_network import add_co2_sequestration_limit, solve_network

logger = logging.getLogger(__name__)


def prepare_network(
    n: pypsa.Network,
    solve_opts: dict,
    planning_horizons: str | None,
    co2_sequestration_potential: dict[str, float],
) -> None:
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    # ADD LOAD_SHEDDING
    # intersect between macroeconomic and surveybased willingness to pay
    # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
    # TODO: retrieve color and nice name from config
    n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
    buses_i = n.buses.index

    # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
    load_shedding = 1e2  # Eur/kWh

    logger.info(f"Adding load shedding with marginal cost {load_shedding} EUR/kWh")
    n.add(
        "Generator",
        buses_i,
        " load",
        bus=buses_i,
        carrier="load",
        sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
        marginal_cost=load_shedding,  # Eur/kWh
        p_nom=1e9,  # kW
    )

    if solve_opts.get("curtailment_mode"):
        n.add("Carrier", "curtailment", color="#fedfed", nice_name="Curtailment")
        n.generators_t.p_min_pu = n.generators_t.p_max_pu
        buses_i = n.buses.query("carrier == 'AC'").index
        n.add(
            "Generator",
            buses_i,
            suffix=" curtailment",
            bus=buses_i,
            p_min_pu=-1,
            p_max_pu=0,
            marginal_cost=-0.1,
            carrier="curtailment",
            p_nom=1e6,
        )

    if n.stores.carrier.eq("co2 sequestered").any():
        limit_dict = co2_sequestration_potential
        add_co2_sequestration_limit(
            n, limit_dict=limit_dict, planning_horizons=planning_horizons
        )

    # Clear old global_constraints for custom constraints
    custom_global_constraints = ["co2_sequestration_min", "electrolyser_capacity_min", "h2_production_min"]
    for cgc in custom_global_constraints:
        if cgc in n.global_constraints.index:
            logger.info(f"Removing global constraint {cgc} meta data.")
            n.global_constraints.drop(cgc, inplace=True)


def update_dict(original_dict: dict, new_dict: dict, depth: int = 0) -> dict:
    """
    Recursively updates the original dictionary with the new dictionary.
    Adds leading dots to the log message based on recursion depth.
    """
    prefix = "." * (depth * 2)  # Two dots per depth level

    for key, value in new_dict.items():
        logger.info(f"{prefix}Updating key: ['{key}'] with value: ['{value}']")
        if isinstance(value, dict) and isinstance(original_dict.get(key), dict):
            original_dict[key] = update_dict(original_dict.get(key, {}), value, depth + 1)
        else:
            original_dict[key] = value
    
    return original_dict


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_sector_network",
            opts="",
            clusters="70",
            ll="v1.05",
            sector_opts="",
            planning_horizons="2030",
            column="ops_ghg_target_same_infra",
            run="pcipmi",
            configfiles=["config/investment-all-targets.config.yaml"]
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.options
    solve_opts["load_shedding"] = True # load shedding always on
    planning_horizons = snakemake.wildcards.get("planning_horizons", None)
    column = snakemake.wildcards.get("column", None)
    solve_operations_col = snakemake.params.solve_operations["definitions"][column]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    logger.info("Fixing optimal capacities.")
    n.optimize.fix_optimal_capacities()

    # Initialise n.params and n.config
    n.params = snakemake.params
    n.config = snakemake.config

    # Update config here
    if solve_operations_col.get("options", {}):
        if solve_operations_col["options"].get("remove_co2_pipelines", False):
            logger.info("Deactivating CO2 pipelines.")
            if "CO2 pipeline" in n.links.carrier.values:
                sum_active = n.links.loc[n.links.carrier == "CO2 pipeline", "active"].sum()
                if sum_active == 0:
                    logger.info("No active CO2 pipeline.")
                n.links.loc[n.links.carrier == "CO2 pipeline", "active"] = False
                logger.info(f" --> Deactivated {sum_active} CO2 pipelines.")
            else: 
                logger.warning("No CO2 pipelines found in network.")
        if solve_operations_col["options"].get("remove_h2_pipelines", False):
            logger.info("Deactivating H2 pipelines.")
            if "H2 pipeline" in n.links.carrier.values:
                sum_active = n.links.loc[n.links.carrier == "H2 pipeline", "active"].sum()
                if sum_active == 0:
                    logger.info("No active CO2 pipeline.")
                n.links.loc[n.links.carrier == "H2 pipeline", "active"] = False
                logger.info(f" --> Deactivated {sum_active} H2 pipelines.")
            else: 
                logger.warning("No H2 pipelines found in network.")

    if solve_operations_col.get("overwrite_config", {}):
        logger.info("Updating config with column-specific options.")
        n.config = update_dict(
            n.config, solve_operations_col["overwrite_config"]
        )
    
    prepare_network(
        n,
        solve_opts=snakemake.params.solving["options"],
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
    )

    logger.info("---")
    logger.info(f"Running operational optimisation for scenario ['{snakemake.wildcards.run}'], column ['{column}'] and year ['{planning_horizons}']")

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        solve_network(
            n,
            config=snakemake.config,
            params=snakemake.params,
            solving=snakemake.params.solving,
            planning_horizons=planning_horizons,
            rule_name=snakemake.rule,
            dispatch_only=True,
        )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])