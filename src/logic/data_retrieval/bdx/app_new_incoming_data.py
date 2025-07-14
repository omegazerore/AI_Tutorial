"""Module for processing competitor zip files, extracting product data, and saving results.

This script searches for zip files in a designated data directory, extracts JSON files from
each zip, processes them into DataFrames, aggregates the results, and saves to CSV.
Designed for competitor analysis based on product GTINs and dates.

Typical usage:
    python app_new_incoming_data.py
"""


import json
import logging
import os
import re
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Optional, List
from zipfile import BadZipFile

import pandas as pd

from src.io.path_definition import get_datafetch


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


class Constants(str, Enum):
    """Constant values used throughout the data processing pipeline."""
    GTIN = "gtin"
    DATE = "date"
    FILENAME = 'new_competitor_product.csv'


def ensure_data_dir(data_dir: str) -> bool:
    """Ensure the existence of the data directory.

    Args:
        data_dir: The directory path to check/create.

    Returns:
        bool: True if the directory exists or was created; False on failure.
    """
    if not os.path.isdir(data_dir):
        try:
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")
            return False
    return True


def get_zip_files(data_dir: str) -> List[str]:
    """List all zip files in the data directory.

    Args:
        data_dir: The directory to search.

    Returns:
        List[str]: Filenames of all zip files in the directory.
    """
    return [
        file for file in os.listdir(data_dir)
        if file.endswith('.zip')
    ]


def extract_and_process_zip(zip_path: str, data_dir: str) -> Optional[pd.DataFrame]:
    """Extracts JSON files from a zip, processes them, and returns combined DataFrame.

    Args:
        zip_path: Filename of the zip file to process.
        data_dir: Directory containing the zip file.

    Returns:
        Optional[pd.DataFrame]: Combined DataFrame of extracted data, or None on failure.
    """
    date = extract_date(Path(zip_path).stem)
    dfs = []
    try:
        with zipfile.ZipFile(os.path.join(data_dir, zip_path), 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    zip_ref.extractall(temp_dir)
                except BadZipFile as e:
                    logger.warning(f"{zip_path}: {e}")
                    return None
                json_paths = [
                    os.path.join(temp_dir, f)
                    for f in os.listdir(temp_dir)
                    if f.endswith('.json')
                ]
                with ProcessPoolExecutor() as executor:
                    futures = [executor.submit(process_json_file, path) for path in json_paths]
                    for future in as_completed(futures):
                        df = future.result()
                        if df is not None:
                            dfs.append(df)
        if not dfs:
            return None
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.dropna(subset=[Constants.GTIN.value], inplace=True)
        combined_df.drop_duplicates(subset=[Constants.GTIN.value], inplace=True)
        combined_df[Constants.DATE.value] = date
        return combined_df
    except Exception as e:
        logger.error(f"Failed processing {zip_path}: {e}")
        return None


def extract_date(filename: str):
    """Extracts date in the format 'YYYY_MM_DD' from the filename.

    Args:
        filename: The filename (without extension).

    Returns:
        Optional[str]: Extracted date string, or None if not found.
    """
    match = re.search(r'(\d{4}_\d{2}_\d{2})$', filename)
    if match:
        return match.group(1)
    return None


def process_json_file(json_path: str) -> Optional[pd.DataFrame]:
    """Reads a JSON file and returns a DataFrame.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Optional[pd.DataFrame]: DataFrame with data if successful, else None.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
            df = pd.DataFrame(data)
            return df
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to process {json_path}: {e}")
    return None


def aggregate_dataframes(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Aggregates multiple DataFrames, sorts, drops duplicates, and pivots on date and GTIN.

    Args:
        dfs: List of DataFrames to aggregate.

    Returns:
        Optional[pd.DataFrame]: Aggregated DataFrame, or None if input is empty.
    """
    if not dfs:
        logger.warning("No data frames to process.")
        return None
    df = pd.concat(dfs)
    df.sort_values(by=Constants.DATE.value, inplace=True)
    df.dropna(subset=[Constants.GTIN.value], inplace=True)
    df.drop_duplicates(subset=[Constants.GTIN.value], inplace=True)
    df = df.pivot_table(index=Constants.DATE.value, aggfunc={Constants.GTIN.value:"count"})
    return df


def save_output(df: pd.DataFrame, data_dir: str, filename: str) -> None:
    """Saves the DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        data_dir: Directory to save the file in.
        filename: Name of the output CSV file.
    """
    output_file = os.path.join(data_dir, filename)
    df.to_csv(output_file)
    logger.info(f"Output saved to {output_file}")


def main() -> int:
    """Main processing pipeline.

    Processes competitor zip files, extracts JSON data, aggregates results, and writes to a CSV.

    Returns:
        int: 0 if successful, 1 otherwise.
    """
    data_dir = get_datafetch()
    if not ensure_data_dir(data_dir):
        return 1
    zip_files = get_zip_files(data_dir)
    output_dfs = []
    for zip_path in zip_files:
        result = extract_and_process_zip(zip_path, data_dir)
        if result is not None:
            output_dfs.append(result)

    df = aggregate_dataframes(output_dfs)
    if df is None:
        return 1
    save_output(df, data_dir, Constants.FILENAME.value)
    return 0


if __name__ == "__main__":
    main()