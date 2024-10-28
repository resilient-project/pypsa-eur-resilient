# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2021-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Retrieve PCI and PMI project list and codes from PDF annex.

https://energy.ec.europa.eu/document/download/3db5e3d1-9989-4d10-93e3-67f5b9ad9103_en?filename=Annex%20PCI%20PMI%20list.pdf
"""

import logging
import re

import pandas as pd
import pdfplumber
from _helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _extract_tables_from_page(pdf_path, page_number):
    """
    Extracts tables from a specific page of a PDF file.

    Parameters:
        pdf_path (str): The path to the PDF file.
        page_number (int): The page number to extract tables from.
    Returns:
        df (pandas.DataFrame): A DataFrame containing the extracted tables with columns "pci_code" and "definition".
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        tables = page.extract_tables()

    logger.info(
        f"Extracting {len(tables)}"
        + " table"
        + ("s" if len(tables) > 1 else "")
        + f" from page {page_number}."
    )

    # Check if any tables were extracted
    if not tables:
        # Return an empty DataFrame with specified columns
        return pd.DataFrame(columns=["pci_code", "definition"])

    # Convert tables to DataFrames and concatenate them
    df = pd.concat([pd.DataFrame(table) for table in tables], ignore_index=True)
    df.columns = ["pci_code", "definition"]

    # Remove rows where "pci_code" is "No." (header)
    df = df[df["pci_code"] != "No."]

    return df


def _extract_all_tables(pdf_path, page_numbers):
    """
    Extracts tables from all pages of a PDF file and concatenates them. Forward
    fills the "pci_code" column and aggregates the "definition" column by
    concatenating the strings, if table is flowing over multiple pages.

    Parameters:
    - pdf_path (str): The path to the PDF file.
    - page_numbers (list): A list of page numbers to extract tables from.
    Returns:
    - df (pandas.DataFrame): A DataFrame containing the extracted tables, with columns "pci_code" and "definition".
    """
    logger.info(
        "Extracting PCI/PMI project tables from pages "
        + ", ".join(map(str, page_numbers))
        + f" in file:\n{pdf_path}\n"
    )
    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=["pci_code", "definition"])

    # Extract tables from each page and concatenate them
    for page_number in page_numbers:
        df = pd.concat(
            [df, _extract_tables_from_page(pdf_path, page_number)], ignore_index=True
        )

    # where df["pci_code"]=="" assign the value of the previous row
    df["pci_code"] = df["pci_code"].replace("", pd.NA)
    df["pci_code"] = df["pci_code"].ffill()

    # Group by "pci_code" and aggregate "definition" column by concatenating the strings
    df = df.groupby("pci_code")["definition"].agg(" ".join).reset_index()

    return df


def _get_contained_projects(pci_code, definition):
    """
    Get the list of contained projects/subprojects based on the given PCI code
    and definition.

    Parameters:
        pci_code (str): The PCI code to match.
        definition (str): The definition string to search for matching numeric strings that start with the PCI code.
    Returns:
        list (list): A sorted list of unique projects that match the given PCI code. If no subprojects are found, the list will contain only the PCI code.
    """
    # Regex pattern to match substrings that start with the pci_code
    pattern = rf"{pci_code}\.\d+"

    # Find all matching substrings
    projects = re.findall(pattern, definition)

    # if subprojects is empty, initialize with pci_code
    if not projects:
        projects = [pci_code]

    return sorted(list(set(projects)))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_pci_pmi_list")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    pdf_path = snakemake.input[0]
    project_list = snakemake.output[0]

    df = _extract_all_tables(pdf_path, range(2, 14))
    df["projects"] = df.apply(
        lambda row: _get_contained_projects(row["pci_code"], row["definition"]), axis=1
    )
    list_of_projects = [
        project for project_list in df["projects"] for project in project_list
    ]

    # Drop project with ID "9.19"
    if "9.19" in list_of_projects:
        list_of_projects.remove("9.19")

    logger.info(f"Total number of PCI/PMI projects: {len(list_of_projects)}.\n")

    logger.info(f"Exporting PCI/PMI project list to:\n{project_list}.")
    # Write the list to a text file
    with open(project_list, "w") as f:
        for project in list_of_projects:
            f.write(f"{project}\n")
