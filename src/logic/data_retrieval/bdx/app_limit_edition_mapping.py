import json
import logging
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from zipfile import BadZipFile

import pandas as pd

from src.io.path_definition import get_datafetch


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['gtin', 'limited_edition', 'url']


def process_json_file(json_path: str) -> Optional[pd.DataFrame]:
    """Reads a JSON file and returns a DataFrame if 'limited_edition' column exists.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame or None: DataFrame with data if 'limited_edition' exists, else None.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
            df = pd.DataFrame(data)
            if 'limited_edition' in df.columns:
                return df
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to process {json_path}: {e}")
    return None


def main() -> int:
    """Processes competitor zip files, extracts JSON data with 'limited_edition' entries,
    and writes them to a CSV file.
    """
    data_dir = get_datafetch()
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1

    limited_edition_dfs = []

    for zip_path in os.listdir(data_dir):
        if not zip_path.endswith(".zip"):
            continue

        with zipfile.ZipFile(os.path.join(data_dir, zip_path), 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    zip_ref.extractall(temp_dir)
                except BadZipFile as e:
                    logger.warning(f"{zip_path}: {e}")
                    continue
                json_paths = [
                    os.path.join(temp_dir, filename)
                    for filename in os.listdir(temp_dir)
                    if filename.endswith('.json')
                ]

                with ProcessPoolExecutor() as executor:
                    futures = [executor.submit(process_json_file, path) for path in json_paths]
                    for future in as_completed(futures):
                        df = future.result()
                        if df is not None:
                            limited_edition_dfs.append(df)

    if not limited_edition_dfs:
        logger.warning("No data found with 'limited_edition' column.")
        return 1

    combined_df = pd.concat(limited_edition_dfs, ignore_index=True)
    combined_df.sort_values(by='gtin', inplace=True)

    for col in REQUIRED_COLUMNS:
        if col not in combined_df.columns:
            raise KeyError(f"Missing required column: {col}")

    output_file = os.path.join(data_dir, 'limited_edition_table.csv')

    combined_df[REQUIRED_COLUMNS].to_csv(output_file, index=False)

    return 0


if __name__ == "__main__":
    main()