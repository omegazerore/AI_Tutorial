import argparse
import csv
import logging
import os
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List

from langchain_community.document_loaders import Docx2txtLoader
import mlflow
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import pandas as pd
from pydantic_settings import BaseSettings
from unstructured.partition.pdf import partition_pdf

from src.logic.trendweek_report.constants import Constants
from src.io.path_definition import get_datafetch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PartitionPDFSettings(BaseSettings):
    """
    Configuration settings for partition function parameters.
    """

    CHUNKING_STRATEGY: str = 'by_title'
    INFER_TABLE_STRUCTURE: bool = True
    EXTRACT_IMAGE_BLOCK_TYPES: List[str] = ["Image"]
    MAX_CHARACTERS: int = 4000
    NEW_AFTER_N_CHARS: int = 3800
    COMBINE_TEXT_UNDER_N_CHARS: int = 2000
    STRATEGY: str = 'hi_res'


settings = PartitionPDFSettings()


def safe_parse_pdf(path, filename):
    """Safely parses a PDF file into text chunks using the Unstructured partitioner.

    Args:
        path (str): Path to the PDF file.
        filename (str): Name of the file.

    Returns:
        List[tuple]: List of (text, filename) tuples.
    """
    elements = partition_pdf(path,
                             chunking_strategy=settings.CHUNKING_STRATEGY,
                             infer_table_structure=settings.INFER_TABLE_STRUCTURE,
                             extract_image_block_types=settings.EXTRACT_IMAGE_BLOCK_TYPES,
                             max_characters=settings.MAX_CHARACTERS,
                             new_after_n_chars=settings.NEW_AFTER_N_CHARS,
                             combine_text_under_n_chars=settings.COMBINE_TEXT_UNDER_N_CHARS,
                             extract_image_block_output_dir=f"{Path(filename).stem}/figures",
                             strategy=settings.STRATEGY)
    return [(e.text, filename) for e in elements]


def safe_parse_word(path, filename):
    """Safely parses a Word (.docx) file into text chunks.

    Args:
        path (str): Path to the Word file.
        filename (str): Name of the file.

    Returns:
        List[tuple]: List of (text, filename) tuples.
    """
    loader = Docx2txtLoader(path)
    elements = loader.load()
    return [(e.text, filename) for e in elements]


def append_to_csv(output_filename, rows, header=False):
    """Appends parsed content rows to a CSV file.

    Args:
        output_filename (str): Target CSV file.
        rows (List[tuple]): Rows of (text, filename) to write.
        header (bool, optional): Whether to write headers. Defaults to False.
    """
    with open(output_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(['text', 'file'])
        for row in rows:
            if row:  # Avoid None
                writer.writerow(row)


def parse_file(path_tuple):
    """Parses a single file (PDF or Word) and returns parsed rows.

    Args:
        path_tuple (tuple): Tuple of (path, filename, is_pdf).

    Returns:
        List[tuple]: List of parsed (text, filename) tuples.
    """
    path, filename, is_pdf = path_tuple
    try:
        if is_pdf:
            try:
                return safe_parse_pdf(path, filename)
            except Exception as e:
                logger.exception(f"Error parsing PDF {filename}")
                return []
        else:
            try:
                return safe_parse_word(path, filename)
            except Exception as e:
                logger.exception(f"Error parsing WORD {filename}")
                return []
    except Exception as e:
        return []


def resolve_output_path(args) -> str:
    """Generates the appropriate output CSV path.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        str: Output file path.
    """

    if args.test:
        return os.path.join(get_datafetch(), *(["report_raw.csv"] if args.scope == "future_vision"
                                                else [args.source, "report_raw_test.csv"]))
    return os.path.join(get_datafetch(), *(["report_raw.csv"] if args.scope == "future_vision"
                                            else [args.source, "report_raw.csv"]))


def get_files_to_process(directory: str, already_processed: List[str],
                         test_limit: int = None) -> List[tuple]:
    """Gets a list of files to parse from the directory, excluding already processed ones.

    Args:
        directory (str): Directory path to scan.
        already_processed (List[str]): List of filenames already parsed.
        test_limit (int, optional): Max number of files for test mode. Defaults to None.

    Returns:
        List[tuple]: List of (path, filename, is_pdf) tuples.
    """
    files = [f for f in os.listdir(directory) if f not in already_processed and
             f.endswith(('.pdf', '.docx'))]
    if test_limit:
        files = files[:test_limit]
    return [(os.path.join(directory, f), f, f.endswith('.pdf')) for f in files]


def process_file_tasks(file_tasks: List[tuple], output_filename: str, logger) -> int:
    """Processes all file parsing tasks in parallel.

    Args:
        file_tasks (List[tuple]): Files to parse.
        output_filename (str): Output CSV filename.
        logger (Logger): Logger instance.

    Returns:
        int: Number of files successfully processed.
    """
    counter = 0
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(parse_file, task): task[1] for task in file_tasks}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file)):
            filename = future_to_file[future]
            try:
                rows = future.result()
                append_to_csv(output_filename, rows)
                logger.info(f"{filename} parsed and written.")
                counter += 1
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
    return counter


def process_files_parallel(args, logger):
    """Main file processing pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments.
        logger (Logger): Logger instance.

    Returns:
        tuple: Number of files processed and total duration.
    """
    directory = os.path.join(get_datafetch(), *(["report"] if args.scope == "future_vision"
                                          else [args.source, "report"]))

    if not os.path.isdir(directory):
        logger.info(f"folder {directory} is not found")
        sys.exit()

    output_filename = resolve_output_path(args)

    if os.path.isfile(output_filename):
        df = pd.read_csv(output_filename)
        processed_files = list(df['file'].unique())
    else:
        append_to_csv(output_filename, [], header=True)
        processed_files = []

    test_limit = None

    if args.test:
        test_limit = Constants.TEST_LIMIT

    file_tasks = get_files_to_process(directory=directory, already_processed=processed_files,
                                      test_limit=test_limit)

    begin = datetime.now()

    counter = process_file_tasks(file_tasks, output_filename, logger)

    end = datetime.now()

    total_time = end - begin

    return counter, total_time


def main(args):
    """Main MLFlow logging and document parsing orchestration.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    logger.info(f"Start {args.scope} report step1")

    platform = sys.platform

    if platform == 'linux':
        mlflow.set_tracking_uri(uri=os.environ['MLFLOW_URL'])
        mlflow.set_experiment(os.environ['WORKFLOWNAME'])
    else:
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        experiment_name = f"trendweek_report_{args.scope}" if args.scope == Constants.FUTURE_VISION else f"trendweek_report_{args.scope}_{args.source}"
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=os.environ['WORKFLOWRUN']) as run:
        run_id = run.info.run_id
        save_run_id(run_id, args)

        logger.info(f"datafetch: {get_datafetch()}")

        counter, total_time = process_files_parallel(args, logger)

        if args.scope in ['future_vision']:
            logger.info(f"{args.scope}_report_step1 total time: {total_time}")
        else:
            logger.info(f"{args.scope}_{args.source}_report_step1 total time: {total_time}")

        if args.scope in ['future_vision']:
            run_name = f"{args.scope}_report_step1"
        else:
            run_name = f"{args.scope}_{args.source}_report_step1"

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_metric("number of documents", counter)
            mlflow.log_param("test", int(args.test))
            mlflow.log_metric("Time", total_time.seconds)
            mlflow.log_metric("processed files", counter)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Flag to enable debugging')
    parser.add_argument('--scope', type=str, help='project')
    parser.add_argument('--source', type=str, required=False, help='source of data. necessary for project `strategy_input`')
    args = parser.parse_args()

    if args.scope not in ['future_vision', 'strategy_input']:
        print(f"{args.scope} is not supported")
        sys.exit()

    if args.scope == 'strategy_input':
        if args.source is None:
            print(f"{args.scope} needs argument `source`")
            sys.exit()

    dir = os.path.join(get_datafetch(), *(["report"] if args.scope == "future_vision"
                                          else [args.source, "report"]))

    if not os.path.isdir(dir):
        os.makedirs(dir)
        sys.exit(f"directory {dir} does not exist")
    else:
        logger.info(f"directory: {dir}")

    main(args=args)