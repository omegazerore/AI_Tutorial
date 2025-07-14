"""Script for processing and cleansing GTIN-tagged product data from ZIP archives.

This includes data extraction from JSON files, optional caching, and applying a
GPT-based cleansing pipeline with MLflow tracking.
"""

import argparse
import os
import json
import logging
import re
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, List, Any
from zipfile import BadZipFile

import mlflow
import pandas as pd

from src.api import Constants
from src.initialization import model_activation
from src.io.path_definition import get_datafetch
from src.logic.websearch.prompt_template import (
    pydantic2text,
    extraction_prompt,
    websearch,
    output_parser
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TAGGED_CLASS_COL = 'tagged'


def extract_retailer(filename):
    """Extracts the retailer name from the filename.

    Args:
        filename (str): Name of the file.

    Returns:
        Optional[str]: Retailer identifier if matched, otherwise None.
    """
    match = re.match(r'^([a-z0-9_]+?)_\d{14}__\d+_\d+_\d+$', filename)
    if match:
        return match.group(1)
    return None


def safe_listdir(path: str) -> List[str]:
    """Safely lists files in a directory with error handling.

    Args:
        path (str): Directory path.

    Returns:
        List[str]: List of file names, empty if error occurs.
    """
    try:
        return os.listdir(path)
    except FileNotFoundError:
        logger.error(f"Directory does not exist: {path}")
    except PermissionError:
        logger.error(f"Permission denied for: {path}")
    return []


def process_json_file(json_path: str) -> Optional[pd.DataFrame]:
    """Processes a JSON file and returns a filtered DataFrame.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        Optional[pd.DataFrame]: DataFrame if valid, otherwise None.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
            df = pd.DataFrame(data)
            if Constants.GTIN.value not in df.columns:
                return None
            df.dropna(subset=[Constants.GTIN.value], inplace=True)
            df[Constants.GTIN.value] = df[Constants.GTIN.value].astype(str)
            df.drop_duplicates(subset=[Constants.GTIN.value], inplace=True)
            df[Constants.RESELLER.value] = extract_retailer(Path(json_path).stem)
            if 'base_name' not in df.columns:
                df['base_name'] = ""
        return df[[Constants.GTIN.value, 'base_name', Constants.RESELLER.value ]]
    except json.JSONDecodeError:
        logger.error(f"Malformed JSON in: {json_path}")
    except IOError:
        logger.error(f"Failed to process {json_path}")
    return None


def prepare_inputs(df_query: pd.DataFrame, test: bool) -> List[Dict[str, Any]]:
    """Prepares input records for GPT batch processing.

    Args:
        df_query (pd.DataFrame): Source query data.
        test (bool): If True, returns all rows; else only untagged ones.

    Returns:
        List[Dict[str, Any]]: List of dictionaries for model input.
    """
    df_query_copy = df_query.copy()
    df_query_copy.rename(columns={
        Constants.DESCRIPTION.value: "product",
        Constants.RESELLER.value: "retailer"
    }, inplace=True)

    data = df_query_copy.iterrows()
    if test:
        return [row.to_dict() for _, row in data][::-1]
    else:
        return [row.to_dict() for _, row in data if not row[TAGGED_CLASS_COL]][::-1]


def get_or_load_cached_csv(zip_name: str, data_dir: str, re_run: bool) -> Optional[pd.DataFrame]:
    """Loads cached CSV if available and re_run is False.

    Args:
        zip_name (str): ZIP file name.
        data_dir (str): Path to data directory.
        re_run (bool): Whether to bypass cache.

    Returns:
        Optional[pd.DataFrame]: Cached DataFrame or None.
    """
    csv_path = os.path.join(data_dir, f"{Path(zip_name).stem}.csv")
    if not re_run and os.path.isfile(csv_path):
        logger.info(f"Using cached: {csv_path}")
        return pd.read_csv(csv_path, dtype={'gtin': str})
    return None


def extract_json_from_zip(zip_path: str, temp_dir: str) -> List[str]:
    """Extracts all JSON files from a ZIP archive.

    Args:
        zip_path (str): Path to ZIP file.
        temp_dir (str): Temporary directory to extract files to.

    Returns:
        List[str]: Paths to extracted JSON files.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            return [
                os.path.join(temp_dir, f)
                for f in os.listdir(temp_dir) if f.endswith('.json')
            ]
    except BadZipFile as e:
        logger.warning(f"{zip_path}: {e}")
        return []


def parse_zip_file(zip_name: str, data_dir: str) -> pd.DataFrame:
    """Parses a ZIP file and extracts JSON content into a DataFrame.

    Args:
        zip_name (str): ZIP file name.
        data_dir (str): Directory containing the ZIP file.

    Returns:
        pd.DataFrame: Combined DataFrame of extracted JSON data.
    """
    zip_path = os.path.join(data_dir, zip_name)
    dfs_zip = []

    with tempfile.TemporaryDirectory() as temp_dir:
        json_files = extract_json_from_zip(zip_path, temp_dir)

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_json_file, path) for path in json_files]
            for future in as_completed(futures):
                df = future.result()
                if df is not None:
                    dfs_zip.append(df)

    if dfs_zip:
        df = pd.concat(dfs_zip, ignore_index=True).drop_duplicates(subset=['gtin'])
        df.rename(columns={'base_name': Constants.DESCRIPTION.value}, inplace=True)
        return df
    return pd.DataFrame()


def log_and_exit(msg: str, exception: Optional[Exception] = None, exit_code: int = 1):
    """Logs an error and exits the script.

    Args:
        msg (str): Log message.
        exception (Optional[Exception]): Exception object.
        exit_code (int): Exit code.
    """
    if exception:
        logger.exception(msg)
    else:
        logger.error(msg)
    sys.exit(exit_code)


def data_preparation(args: argparse.Namespace) -> pd.DataFrame:
    """Main function for loading and preparing product data.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        pd.DataFrame: Combined DataFrame of all valid product records.
    """
    data_dir = get_datafetch()
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        log_and_exit(f"Data directory missing: {data_dir}")

    all_dfs = []

    logger.info(f"re_run = {args.re_run}")

    for zip_file in safe_listdir(data_dir):
        if not zip_file.endswith(".zip"):
            continue

        cached_df = get_or_load_cached_csv(zip_file, data_dir, args.re_run)
        if cached_df is not None:
            all_dfs.append(cached_df)
            continue

        logger.info(f"Parsing ZIP: {zip_file}")
        df = parse_zip_file(zip_file, data_dir)

        if not df.empty:
            out_path = os.path.join(data_dir, f"{Path(zip_file).stem}.csv")
            df.to_csv(out_path, index=False)
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['gtin'])


def safe_to_csv(df: pd.DataFrame, path: str) -> None:
    """Safely saves a DataFrame to CSV with exception handling.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Output path.

    Raises:
        Exception: On failure to write.
    """
    try:
        df.to_csv(path, index=False)
    except (IOError, OSError) as e:
        logger.exception(f"Failed to write to {path}")
        raise


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_search', type=int, help='number of data to be cleaned')
    parser.add_argument('--re_run', action='store_true', help='if we are going to re-run through all the .zip files')
    parser.add_argument('--test', action='store_true',
                        help='Enable test mode. Processes only the first 2000 records and saves results to a test output file.')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of items to process in each batch.')
    return parser.parse_args()


def setup_mlflow(run_name: Optional[str] = None) -> str:
    """Configures MLflow tracking URI and returns run name.

    Args:
        run_name (Optional[str]): Optional run name override.

    Returns:
        str: MLflow run name.
    """
    platform = sys.platform
    if platform == 'linux':
        run_name = os.getenv('WORKFLOWNAME', 'default_run')
        tracking_uri = os.getenv('MLFLOW_URL', 'http://127.0.0.1:8080')
        mlflow.set_tracking_uri(uri=tracking_uri)
        logger.info(f"architecture build: {os.environ['BASE_TAG']}")
    else:
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("app_bdx_cleansing")

    return run_name


def build_pipeline(model_name: str = "gpt-4o-mini"):
    """Constructs the transformation pipeline.

    Args:
        model_name (str): Model name for activation.

    Returns:
        Any: Composed pipeline function.
    """
    model = model_activation(model_name=model_name)
    prompt_template = extraction_prompt()
    return {"text": websearch} | prompt_template | model | output_parser | pydantic2text


def main():
    """Main entry point of the script. Runs full pipeline."""
    args = parse_args()

    output_filename = os.path.join(get_datafetch(), 'groupbwt_raw_unique_gtin_eansearch_cleaned.csv')
    run_name = setup_mlflow()

    df_source = data_preparation(args)

    if os.path.isfile(output_filename):
        logger.info(f"{output_filename} exists.")
        df_cleaned = pd.read_csv(output_filename, dtype={Constants.GTIN.value: 'str'})
        df_cleaned.dropna(subset=[Constants.GTIN.value], inplace=True)
        df_cleaned.drop_duplicates(subset=['gtin'], inplace=True)
        # update df
        df_uncleaned = df_source[~df_source[Constants.GTIN.value].isin(df_cleaned[Constants.GTIN.value].tolist())]
        # concatenate cleaned data from df
        df = pd.concat([df_cleaned, df_uncleaned], ignore_index=True)
        df[TAGGED_CLASS_COL].fillna(False, inplace=True)
    else:
        logger.info(f"{output_filename} does not exist.")
        df = df_source.copy()
        df.loc[:, 'cleaned_product_name'] = None
        df.loc[:, TAGGED_CLASS_COL] = False

    pipeline_ = build_pipeline(model_name="gpt-4o-mini")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metric("n_cleaned", df[TAGGED_CLASS_COL].sum())
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("mlflow_run_id", run.info.run_id)
        mlflow.log_param("platform", sys.platform)
        if sys.platform == 'linux':
            mlflow.log_param("base_tag", os.environ['BASE_TAG'])
            mlflow.log_param("worflowname", os.environ['WORKFLOWNAME'])
            mlflow.log_param("run_id", f"{os.environ['WORKFLOWNAME']}-{os.environ['BASE_TAG']}")
            mlflow.log_param("namespace", os.environ['WORKFLOWNAMESPACE'])

        while len(df[df[TAGGED_CLASS_COL] == False]) > 0:
            inputs_ = prepare_inputs(df, test=args.test)
            try:
                df.set_index([Constants.GTIN.value], inplace=True)
            except KeyError as e:
                logger.exception(f"Missing expected {Constants.GTIN.value} column.")
                raise ValueError("GTIN column is required in the dataframe.") from e

            batch = [inputs_.pop() for _ in range(min(args.batch_size, len(inputs_)))]
            gtins = [item[Constants.GTIN.value] for item in batch]
            outputs = pipeline_.batch(batch)

            try:
                df.loc[gtins, "cleaned_product_name"] = outputs
            except ValueError as e:
                logger.error(f"size of gtins: {len(gtins)}")
                logger.error(f"size of matched dataframe: {len(df.loc[gtins])}")
                sys.exit(1)

            df.loc[gtins, TAGGED_CLASS_COL] = True
            df.reset_index(inplace=True)

            if args.test:
                with mlflow.start_run(run_name=run_name):
                    for element, output in zip(batch, outputs):
                        mlflow.log_param(f"{element[Constants.GTIN.value]} {element['retailer']}", output)
                sys.exit("End of test")

            safe_to_csv(df, output_filename)

if __name__ == "__main__":
    """
    To run MLflow server:
        $ mlflow server --host 127.0.0.1 --port 8080
    """

    main()

