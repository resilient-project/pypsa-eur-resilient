# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Extract optimal link capacities from a solved PyPSA network.
"""

import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "extract_optimal_link_capacities",
            clusters="adm",
            opts="",
            sector_opts="",
            planning_horizons="2040",
            configfiles=["config/pcipmi.config.yaml"],
            run="central-planning",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)

    optimal_p_nom = n.links.query("carrier=='H2 pipeline' or carrier=='CO2 pipeline'")[["carrier", "p_nom_opt", "p_nom", "p_nom_max", "p_nom_extendable", "active"]]

    # Export
    optimal_p_nom.to_csv(snakemake.output[0])

   