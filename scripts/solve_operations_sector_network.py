# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch using the capacities of previous capacity expansion in rule :mod:`solve_network`.
Custom constraints and extra_functionality can be set in the config.
"""

import logging

import numpy as np
import pandas as pd
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
    buses_i = pd.Index(n.loads.bus.unique())
    buses_i = buses_i[~buses_i.str.contains("process emissions")]

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

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

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
            planning_horizons="2050",
            column="ops___ghg_h2_target___no_pipes_short_term_invest",
            run="pcipmi",
            configfiles=["config/investment-all-targets.config.yaml"]
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.options
    solve_opts["load_shedding"] = True # load shedding always on
    
    resample_temporal_resolution = snakemake.params.solve_operations.get("resample_temporal_resolution", False)
    if resample_temporal_resolution:
        logger.info(f"Resampling temporal resolution to {resample_temporal_resolution}h.")
        solve_opts["nhours"] = resample_temporal_resolution    
    
    planning_horizons = snakemake.wildcards.get("planning_horizons", None)
    column = snakemake.wildcards.get("column", None)
    solve_operations_col = snakemake.params.solve_operations["definitions"][column]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    # Initialise n.params and n.config
    n.params = snakemake.params
    n.config = snakemake.config

    # Additional params from solved network:
    emission_price = np.round(abs(n.buses_t.marginal_price.T.loc["co2 atmosphere"].values[0]), 6)

    # Update config here
    if solve_operations_col.get("options", {}):

        # Allow p_nom_extendable / investments:
        if solve_operations_col["options"].get("p_nom_extendable", False):
            logger.info("Allowing investments. Setting p_nom_min to p_nom_opt")
            
            comps = ["Generator", "Link", "StorageUnit", "Store"]
            for c in comps:
                ext_i = n.get_extendable_i(c)
                attrs = n.component_attrs[c]
                nominal_attr = attrs.loc[attrs.index.str.endswith("_nom")].index.values[0]

                if ext_i.any():
                    n.static(c).loc[ext_i, nominal_attr+"_min"] = n.static(c).loc[ext_i, nominal_attr+"_opt"]

            # Disabling extendability of lines
            n.lines.s_nom_extendable = False

            # Disabling extendability of CO2 pipelines and H2 pipelines if they are present
            if "CO2 pipeline" in n.links.carrier.values:
                logger.info("Disabling extendability of CO2 pipelines.")
                n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom"] = n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom_opt"]
                n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom_extendable"] = False
            
            if "H2 pipeline" in n.links.carrier.values:
                logger.info("Disabling extendability of H2 pipelines.")
                n.links.loc[n.links.carrier == "H2 pipeline", "p_nom"] = n.links.loc[n.links.carrier == "H2 pipeline", "p_nom_opt"]
                n.links.loc[n.links.carrier == "H2 pipeline", "p_nom_extendable"] = False

            dispatch_only=False
        else:
            logger.info("Fixing optimal capacities. Not allowing investments.")
            n.optimize.fix_optimal_capacities()
            dispatch_only=True

        # Remove CO2 pipelines
        if solve_operations_col["options"].get("remove_co2_pipelines", False):
            logger.info("Deactivating CO2 pipelines.")
            if "CO2 pipeline" in n.links.carrier.values:
                sum_active = n.links.loc[n.links.carrier == "CO2 pipeline", "active"].sum()
                if sum_active == 0:
                    logger.info("No active CO2 pipeline.")
                n.links.drop(n.links.loc[n.links.carrier == "CO2 pipeline"].index, inplace=True)
                logger.info(f" --> Dropped {sum_active} CO2 pipelines.")
                n.carriers.drop("CO2 pipeline", inplace=True)
            else: 
                logger.warning("No CO2 pipelines found in network.")
        
        # Remove H2 pipelines
        if solve_operations_col["options"].get("remove_h2_pipelines", False):
            logger.info("Deactivating H2 pipelines.")
            if "H2 pipeline" in n.links.carrier.values:
                sum_active = n.links.loc[n.links.carrier == "H2 pipeline", "active"].sum()
                if sum_active == 0:
                    logger.info("No active CO2 pipeline.")
                n.links.drop(n.links.loc[n.links.carrier == "H2 pipeline"].index, inplace=True)
                logger.info(f" --> Dropped {sum_active} H2 pipelines.")
                n.carriers.drop("H2 pipeline", inplace=True)
            else: 
                logger.warning("No H2 pipelines found in network.")

        # Remove CO2 sequestration sites
        if solve_operations_col["options"].get("remove_co2_sequestration_sites", False):
            logger.info("Deactivating CO2 sequestration sites.")
            if "co2 sequestered" in n.stores.carrier.values:
                sum_active = n.stores.loc[n.stores.carrier == "co2 sequestered", "active"].sum()
                if sum_active == 0:
                    logger.info("No active CO2 sequestration sites.")
                n.stores.loc[n.stores.carrier == "co2 sequestered", "active"] = False
                logger.info(f" --> Deactivated {sum_active} CO2 sequestration sites.")

                # Dropping links to CO2 sequestration sites
                n.links.drop(n.links.loc[n.links.carrier=="co2 sequestered"].index, inplace=True)
                n.carriers.drop("co2 sequestered", inplace=True)

        co2_atmosphere_constraint = True # Default is true
        if solve_operations_col["options"].get("use_emission_price", False):
            logger.info(f"Using emission price of {emission_price} EUR/t CO2/h from investment run.")
            logger.info(f"Disabling CO2 amosphere constraint.")
            co2_atmosphere_constraint = False

            # TODO: add emission price
            # TODO: objective function is wrong, sometimes negative
            # TODO check vent
            # TODO check 

        # # Add CO2 emission shedding
        # co2_shedding = solve_operations_col["options"].get("co2_shedding", False)
        # if co2_shedding:
        #     logger.info(f"Adding CO2 emissiong shedding at a price of {co2_shedding} EUR/t CO2/h.")
        #     n.add(
        #         "Generator",
        #         "co2 shedding",
        #         bus="co2 atmosphere",
        #         carrier="co2",
        #         marginal_cost=co2_shedding, # EUR/t CO2/h
        #         p_nom=1e6,  #t CO2/h
        #         p_min_pu=-1,
        #         p_max_pu=0,
        #     )

    if solve_operations_col.get("overwrite_config", {}):
        logger.info("Updating config with column-specific options.")
        n.config = update_dict(
            n.config, solve_operations_col["overwrite_config"]
        )
    
    prepare_network(
        n,
        solve_opts=solve_opts,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
    )

    logger.info("---")
    logger.info(f"Running operational optimisation for scenario ['{snakemake.wildcards.run}'], column ['{column}'] and year ['{planning_horizons}']")

    buses0 = n.buses.query("carrier=='co2 stored'").index
    # n.add(
    #     "Link",
    #     buses0, 
    #     suffix="vents",
    #     bus0=list(buses0),
    #     bus1="co2 atmosphere",
    #     carrier="co2 vent",
    #     p_nom=1000000,
    # )
    # n.add(
    #     "Generator",
    #     buses0,
    #     suffix=" co2_shedding",
    #     bus=buses0,
    #     p_min_pu=0,
    #     p_max_pu=1,
    #     marginal_cost=10000,
    #     carrier="co2_shedding",
    #     p_nom=1e9,
    # )

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
            co2_atmosphere_constraint=co2_atmosphere_constraint,
            dispatch_only=dispatch_only,
        )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])