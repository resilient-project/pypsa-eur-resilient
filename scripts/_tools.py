# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pypsa
import warnings


def update_dict(
    original_dict: dict, 
    new_dict: dict, 
    depth: int = 0,
    enable_logs: bool = True,
) -> dict:
    """
    Recursively updates the original dictionary with the new dictionary.
    Adds leading dots to the log message based on recursion depth.
    """
    prefix = "." * (depth * 2)  # Two dots per depth level

    for key, value in new_dict.items():
        # if depth == 0:
        #     logger.info("Updating config with column-specific options.")
        # logger.info(f"{prefix}Updating key: ['{key}'] with value: ['{value}']")
        if isinstance(value, dict) and isinstance(original_dict.get(key), dict):
            original_dict[key] = update_dict(original_dict.get(key, {}), value, depth + 1)
        else:
            original_dict[key] = value
    
    return original_dict


def fill_missing_carriers(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        new_carriers = set(c.df.carrier.unique()) - set(n.carriers.index)
        if new_carriers:
            n.add("Carrier", list(new_carriers), nice_name=list(new_carriers))


def add_carrier_groups(n, config):

    groups = pd.Series(config["grouping"])
    colors = pd.Series(config["group_colors"])
    n.carriers["group"] = groups.reindex(n.carriers.index, fill_value="")
    n.carriers["group_color"] = n.carriers.group.map(colors).fillna("")

    n.carriers.drop("", inplace=True)
    # if (no_names := n.carriers.query("nice_name == ''").index).any():
    #     warnings.warn(f"Carriers {no_names} have no nice_name")
    # if (no_colors := n.carriers.query("color == ''").index).any():
    #     warnings.warn(f"Carriers {no_colors} have no color")
    if (no_groups := n.carriers.query("group == ''").index).any():
        warnings.warn(f"Carriers {no_groups} have no technology group")
    if (no_group_colors := n.carriers.query("group_color == ''").index).any():
        warnings.warn(f"Carriers {no_group_colors} have no technology group color")
    n.carriers = n.carriers.sort_values(["color"])


def sanitize_locations(n):
    if "EU" not in n.buses.index:
        n.add("Bus", "EU", x=-5.5, y=46)
        n.buses.loc["EU", "location"] = "EU"
        n.buses.loc["co2 atmosphere", "location"] = "EU"
    n.buses["x"] = n.buses.location.map(n.buses.x)
    n.buses["y"] = n.buses.location.map(n.buses.y)


def import_network(path):

    n = pypsa.Network(path)
   
    sanitize_locations(n)
    add_carrier_groups(n, config)
    fill_missing_carriers(n)

    return n