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
from solve_network import solve_network

logger = logging.getLogger(__name__)


def update_dict(
    original_dict: dict, 
    new_dict: dict, 
    depth: int = 0,
) -> dict:
    """
    Recursively updates the original dictionary with the new dictionary.
    Adds leading dots to the log message based on recursion depth.
    """
    prefix = "." * (depth * 2)  # Two dots per depth level

    for key, value in new_dict.items():
        if depth == 0:
            logger.info("Updating config with column-specific options.")
        logger.info(f"{prefix}Updating key: ['{key}'] with value: ['{value}']")
        if isinstance(value, dict) and isinstance(original_dict.get(key), dict):
            original_dict[key] = update_dict(original_dict.get(key, {}), value, depth + 1)
        else:
            original_dict[key] = value
    
    return original_dict


def set_minimum_investment(
    n: pypsa.Network,
    planning_horizons: str,
    comps: list=["Generator", "Link", "Store", "Line"],
) -> None:
    """
    Sets a minimum investment for a given carrier in the network, allows for extendable components of the planning horizon.
    """
    logger.info(f"Fixing optimal capacities for components before the investment run.")
    logger.info("Setting minimum capacities of components based on results from investment run.")
    logger.info(f"Components: {comps}")

    planning_horizons = int(planning_horizons)

    for c in comps:
        ext_i = n.get_extendable_i(c)
        attrs = n.component_attrs[c]
        nominal_attr = attrs.loc[attrs.index.str.endswith("_nom")].index.values[0]

        c_in_build_year = n.static(c).loc[ext_i, "build_year"] == planning_horizons

        # Fix this TODO
        # if ext_i[c_in_build_year].any():
        #     n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_min"] = n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_opt"]
        #     # For case where optimal capacity is slightly higher than maximum capacity due to solver tolerances
        #     n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_max"] = max(
        #         n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_opt"],
        #         n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_max"],
        #     )
        #     if n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_min"] < n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_max"]:
        #         n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_extendable"] = True
        #     if n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_min"] == n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_max"]:
        #         n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_extendable"] = False
        #         n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_nom"] = n.static(c).loc[ext_i[c_in_build_year], nominal_attr+"_opt"]
        
        if ext_i[~c_in_build_year].any():
            n.static(c).loc[ext_i[~c_in_build_year], nominal_attr] = n.static(c).loc[ext_i[~c_in_build_year], nominal_attr+"_opt"]
            n.static(c).loc[ext_i[~c_in_build_year], nominal_attr+"_extendable"] = False


def fix_all_optimal_capacities(
    n: pypsa.Network,
    comps: list=["Generator", "Line", "Link", "Store"],
) -> None:
    """
    Fixes the optimal capacities of extendable components in the network.
    """
    logger.info("Fixing optimal capacities of components based on results from investment run.")
    logger.info(f"Components to fix: {comps}")
    for c in comps:
        ext_i = n.get_extendable_i(c)
        attrs = n.component_attrs[c]
        nominal_attr = attrs.loc[attrs.index.str.endswith("_nom")].index.values[0]

        if ext_i.any():
            n.static(c).loc[ext_i, nominal_attr] = n.static(c).loc[ext_i, nominal_attr+"_opt"]
            n.static(c).loc[ext_i, nominal_attr+"_extendable"] = False


def fix_optimal_pipeline_capacities(
    n: pypsa.Network,
) -> None:
    """
    Fixes the optimal capacities of pipelines in the network.
    """
    logger.info("Fixing optimal capacities of pipelines")
    if "CO2 pipeline" in n.links.carrier.values:
        logger.info("Disabling extendability of CO2 pipelines.")
        n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom"] = n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom_opt"]
        n.links.loc[n.links.carrier == "CO2 pipeline", "p_nom_extendable"] = False
    
    if "H2 pipeline" in n.links.carrier.values:
        logger.info("Disabling extendability of H2 pipelines.")
        n.links.loc[n.links.carrier == "H2 pipeline", "p_nom"] = n.links.loc[n.links.carrier == "H2 pipeline", "p_nom_opt"]
        n.links.loc[n.links.carrier == "H2 pipeline", "p_nom_extendable"] = False


def add_load_shedding(
    n: pypsa.Network,
    marginal_cost: float=10000,
) -> None:
    """
    Adds load shedding to the network.
    """
    n.add("Carrier", "load", color="#dd2e23", nice_name="Load Shedding")
    buses_i = pd.Index(n.loads.bus.unique())

    logger.info(f"Adding load shedding to buses with carriers {n.buses.carrier[buses_i].unique()}.")
    logger.info(f"Load shedding marginal cost: {marginal_cost} EUR/MWh.")
    n.add(
        "Generator",
        buses_i,
        " load",
        bus=buses_i,
        carrier="load",
        marginal_cost=marginal_cost,
        p_nom_extendable=True,
    )    

    n.add(
        "Generator",
        buses_i,
        " load negative",
        bus=buses_i,
        carrier="load",
        marginal_cost=-marginal_cost,
        p_nom_extendable=True,
        p_min_pu=-1,
        p_max_pu=0,
    )    


def remove_pipelines(
    n: pypsa.Network,
    carrier: str,
) -> None:
    """
    Removes carrier pipelines from the network.
    """
    logger.info(f"Removing {carrier}s from the network.")
    if carrier in n.links.carrier.values:
        sum_active = n.links.loc[n.links.carrier == carrier, "active"].sum()
        if sum_active == 0:
            logger.info(f"No active {carrier}s in the network.")
            return
        n.remove("Link", n.links.loc[n.links.carrier == carrier].index)
        logger.info(f"Removed {sum_active} active {carrier}s from the network.")
        n.carriers.drop(carrier, inplace=True)
    else:
        logger.warning(f"No {carrier}s found in the network.")
        return
    

def remove_onshore_pipelines(
    n: pypsa.Network,
    carrier: str,
) -> None:
    """
    Removes onshore carrier pipelines from the network.
    """
    logger.info(f"Removing onshore {carrier}s from the network.")
    if carrier in n.links.carrier.values:
        # TODO: fix by doing clean correction (underwater_fraction) in prepare_sector_network
        b_offshore_link = n.links.underwater_fraction>0
        b_carrier_link = n.links.carrier == carrier
        b_offshore_pci = b_offshore_link & b_carrier_link
        b_offshore_regular = n.links.index.str.contains("co2") & n.links.index.str.contains("offshore") & n.links.index.str.contains("pipeline")
        b_to_drop = ~(b_offshore_pci | b_offshore_regular) & b_carrier_link

        sum_active = n.links.loc[b_to_drop, "active"].sum()
        if sum_active == 0:
            logger.info(f"No active onshore {carrier}s in the network.")
            return
        n.remove("Link", n.links.loc[b_to_drop].index)
        logger.info(f"Removed {sum_active} active onshore {carrier}s from the network.")
        n.carriers.drop(carrier, inplace=True)
    else:
        logger.warning(f"No onshore {carrier}s found in the network.")
        return


def remove_co2_sequestration(
    n: pypsa.Network,
) -> None:
    """
    Removes CO2 sequestration sites from the network.
    """
    logger.info("Removing CO2 sequestration sites from the network.")
    if "co2 sequestered" in n.stores.carrier.values:
        sum_active = n.stores.loc[n.stores.carrier == "co2 sequestered", "active"].sum()
        if sum_active == 0:
            logger.info("No active CO2 sequestration sites.")
        n.remove("Store", n.stores.loc[n.stores.carrier == "co2 sequestered"].index)
        logger.info(f"Removed {sum_active} active CO2 sequestration sites.")

        # Dropping links to CO2 sequestration sites
        n.remove("Link", n.links.loc[n.links.carrier=="co2 sequestered"].index)
        n.carriers.drop("co2 sequestered", inplace=True)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_sector_network",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2040",
            column="ops__no_pipes",
            run="pcipmi-national-international-expansion",
            configfiles=["config/third-run.dev.config.yaml"]
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    rule_name = snakemake.rule
    params = snakemake.params
    config = snakemake.config
    solving = snakemake.params.solving

    planning_horizons = snakemake.wildcards.get("planning_horizons", None)
    column = snakemake.wildcards.get("column", None)
    solve_operations_col = snakemake.params.solve_operations["definitions"][column]

    np.random.seed(solving.get("seed", 123))
    
    # Initialise additional settings to be passed to solve_network function
    additional_settings = dict()
    additional_settings["capacity_constraints"] = False
    additional_settings["co2_atmosphere_constraint"] = True

    n = pypsa.Network(snakemake.input.network)

    # Update solving options, where needed
    solving["options"]["noisy_costs"] = False # Only apply noisy_costs once to enable the same solution space

    #################################
    # COLUMN-SPECIFIC CONFIGURATION #
    #################################

    if solve_operations_col.get("options", {}):
        # Emission price instead of CO2 atmosphere budget
        if solve_operations_col["options"].get("use_emission_price", False):
            ep_factor = solve_operations_col["options"]["use_emission_price"]*1
            emission_price = np.round(abs(n.buses_t.marginal_price.T.loc["co2 atmosphere"].values[0]), 8)
            logger.info(f"Using emission price of {emission_price} x {ep_factor} EUR/t CO2/h from investment run.")
            logger.info(f"Disabling CO2 amosphere constraint.")
            additional_settings["co2_atmosphere_constraint"] = False
            config["costs"]["emission_prices"]["enable"] = True
            config["costs"]["co2"] = emission_price*ep_factor

            # Add generator to co2 atmosphere bus
            n.add(
                "Generator",
                "co2 atmosphere",
                bus="co2 atmosphere",
                p_min_pu=-1,
                p_max_pu=0,
                p_nom_extendable=True,
                marginal_cost=-emission_price*ep_factor,
            )
            additional_settings["empty_co2_atmosphere_store_constraint"] = True

            # Force marginal cost of co2 atmosphere store to 0
            n.stores.loc["co2 atmosphere", "marginal_cost"] = 0
            n.stores.loc["co2 atmosphere", "e_cyclic_per_period"] = False # TODO check if needed
        
        # Removing CO2 pipelines if they are present
        if solve_operations_col["options"].get("remove_co2_pipelines", False):
            remove_pipelines(n, carrier="CO2 pipeline")

        # Removing H2 pipelines if they are present
        if solve_operations_col["options"].get("remove_h2_pipelines", False):
            remove_pipelines(n, carrier="H2 pipeline")

        # Removing CO2 sequestration sites if they are present
        if solve_operations_col["options"].get("remove_co2_sequestration", False):
            remove_co2_sequestration(n)

        if solve_operations_col["options"].get("remove_onshore_co2_pipelines", False):
            remove_onshore_pipelines(n, carrier="CO2 pipeline")

        # Values from previous optimisation run  

        if solve_operations_col["options"].get("allow_investments", False):
            set_minimum_investment(n, planning_horizons)
        if solve_operations_col["options"].get("fix_optimal_capacities", False):
            fix_optimal_pipeline_capacities(n)
        if solve_operations_col["options"].get("fix_all_capacities", False):
            fix_all_optimal_capacities(n)

        # Load shedding
        if solve_operations_col["options"].get("allow_load_shedding", False):
            config["solving"]["options"]["load_shedding"] = True
            marginal_cost = solve_operations_col["options"]["allow_load_shedding"]*1
            add_load_shedding(n, marginal_cost)
        

    #################################

    # Remove minimum part load
    # n.links.loc[n.links.carrier=="Sabatier", "p_min_pu"] = 0
    # n.links.loc[n.links.carrier=="Fischer-Tropsch", "p_min_pu"] = 0
    # n.links.loc[n.links.carrier=="methanolisation", "p_min_pu"] = 0

    # Overwrite individual config options
    if solve_operations_col.get("overwrite_config", {}):
        config = update_dict(
            config, solve_operations_col["overwrite_config"]
        )

    # Store updated params and config in network file
    n.params = params
    n.config = config

    # Run the operational stage of the model
    logger.info("---")
    logger.info(f"Running operational optimisation for column ['{column}'] and year ['{planning_horizons}']")

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        solve_network(
            n,
            config=config,
            params=params,
            solving=solving,
            planning_horizons=planning_horizons,
            rule_name=rule_name,
            additional_settings=additional_settings,
        )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])


# negative_load = (n.generators_t.p.filter(like="load").sum(axis=0) > 0.0)
# negative_load = negative_load[negative_load].index
# n.generators_t.p.T.loc[negative_load]
