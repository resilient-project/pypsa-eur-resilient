# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2021-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Retrieve PCI and PMI projects from interactive PCI-PMI Transparency Platform.

https://ec.europa.eu/energy/infrastructure/transparency_platform/map-viewer/main.html
"""

import json
import logging
import os

import pandas as pd
import requests
from _helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


# Global constants
# Example URL: https://webgate.ec.europa.eu/getis/rest/services/Energy/TP_ELECTRICITY/MapServer/find?searchText=1041&contains=false&searchFields=PCI_ID&layers=5,4,3,2,1&layerDefs={"5":"(PCI_LIST_ACTIVE=1 AND IMPLEMENTATION_STATUS IN(1,2,3,4,5))","4":"(PCI_LIST_ACTIVE=1 AND IMPLEMENTATION_STATUS IN(1,2,3,4,5))","3":"(PCI_LIST_ACTIVE=1 AND IMPLEMENTATION_STATUS IN(1,2,3,4,5))","2":"(PCI_LIST_ACTIVE=1 AND IMPLEMENTATION_STATUS IN(1,2,3,4,5))","1":"(PCI_LIST_ACTIVE=1 AND IMPLEMENTATION_STATUS IN(1,2,3,4,5))"}&f=json
TP_BASE_URL = "https://webgate.ec.europa.eu/getis/rest/services/Energy"
TP_TYPES = [
    "TP_CO2",
    "TP_ELECTRICITY",
    "TP_GAS",
    "TP_HYDROGEN",
    "TP_SMARTGRIDSGAS",
    "TP_SMARTGRIDS",
]
TP_LAYERS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]
TP_LAYER_DEFS = dict(
    {
        **dict.fromkeys(
            TP_LAYERS,
            f"(PCI_LIST_ACTIVE=1 AND IMPLEMENTATION_STATUS IN({','.join(map(str, TP_LAYERS))}))",
        ),
    }
)
TP_FORMAT = "json"
TP_SEARCHFIELD = "PCI_CODE"  # PCI_CODE or PCI_ID
TP_CONTAINS = "False"  # set to false to search for exact match
# Map PCI code prefixes to Transparency Platform (TP) types
PCI_CODE_TYPES = dict(
    {
        **dict.fromkeys(map(str, range(1, 9)), "TP_ELECTRICITY"),
        **dict.fromkeys(map(str, range(9, 12)), "TP_HYDROGEN"),
        "12": "TP_SMARTGRIDS",
        "13": "TP_CO2",
        "14": "TP_SMARTGRIDSGAS",  # empty
        "15": "TP_GAS",
    }
)


def _retrieve_project_data(pci_code):
    """
    Retrieve project data for a given PCI/PMI project code. This function
    constructs a URL based on the provided PCI code and sends a POST request to
    retrieve project data from a specified server. The response is expected to
    be in JSON format and contain a "results" field.

    Parameters:
        pci_code (str): The PCI/PMI project code used to identify the project.

    Returns:
        data (dict): The JSON response from the server if data is found, otherwise None.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request returned an unsuccessful status code.
    """

    project_type = PCI_CODE_TYPES[pci_code.split(".")[0]]
    project_url = (
        TP_BASE_URL
        + "/"
        + project_type
        + "/MapServer/find?searchText="
        + pci_code
        + "&contains="
        + TP_CONTAINS
        + "&searchFields="
        + TP_SEARCHFIELD
        + "&layers="
        + ",".join([str(layer) for layer in TP_LAYERS])
        + "&layerDefs="
        + json.dumps(TP_LAYER_DEFS)
        + "&f="
        + TP_FORMAT
    )

    logger.info(
        f"Retrieving data for PCI/PMI project #{pci_code} from layer {project_type}."
    )

    # Send the request
    response = requests.post(project_url)
    response.raise_for_status()  # Raise HTTPError for bad responses
    data = response.json()["results"]

    if not data:
        logger.info(f"No data found for project {pci_code}")
        return None

    return data


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_pci_pmi_project", pci_code="5.2")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    pci_code = snakemake.params[0]
    data = _retrieve_project_data(pci_code)

    if data:
        filepath = snakemake.output[0]
        with open(filepath, mode="w") as f:
            json.dump(data, f, indent=2)
