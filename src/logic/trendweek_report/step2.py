"""Main module for generating and saving trend report notes.

This script reads input CSVs, constructs prompt templates,
runs them through a language model, and saves the generated notes.
It supports multiple project scopes and integrates with MLflow for logging.

Example:
    python script.py --scope future_vision --test
"""

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime

import mlflow
import pandas as pd
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm

from src.initialization import model_activation
from src.io.path_definition import get_datafetch
from src.logic.trendweek_report.future_vision.constants import Constants
from src.logic.trendweek_report.utils.paths import resolve_report_raw_path
from src.logic.trendweek_report.utils.run_utils import load_run_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompt_template(args: argparse.Namespace):
    """Dynamically import and load the prompt template builder.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (prompt_template, output_parser).
    """
    module_name = f"src.logic.trend_week_report.{args.scope}.note_pipeline"
    spec = importlib.util.find_spec(module_name)
    if not spec:
        logger.error(f"Module '{module_name}' not found.")
        sys.exit(1)
    module = importlib.import_module(module_name)
    return module.build_prompt_template()


def build_batches(df: pd.DataFrame):
    """Group input data into batches by filename.

    Args:
        df: Input dataframe with columns `file` and `text`.

    Returns:
        A list of dictionaries, each containing combined text and the filename.
    """
    batch = []
    filenames = df['file'].unique()
    for filename in tqdm(filenames):
        df_filename = df[df['file'] == filename]
        text = '\n'.join(df_filename['text'].tolist())
        batch.append({"text": text, "filename": filename})
    return batch


def save_results(results, args: argparse.Namespace, ext='json'):
    """Save generated notes to a JSON file.

    Args:
        results: Dictionary mapping filenames to generated results.
        args: Parsed command-line arguments.
        ext: Output file extension.
    """
    output_path = resolve_report_raw_path(args, ext)
    with open(output_path, 'w') as fp:
        json.dump(results, fp)


def log_mlflow_metrics(run_id: str, args: argparse.Namespace, total_time):
    """Log parameters and metrics to MLflow.

    Args:
        run_id: MLflow run ID.
        args: Parsed command-line arguments.
        total_time: Duration of note generation.
    """
    with mlflow.start_run(run_id=run_id):
        run_name = f"{args.scope}_report_step2" if args.scope == Constants.FUTURE_VISION else f"{args.scope}_{args.source}_report_step2"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_param("LLM Engine", args.model_name)
            mlflow.log_param("test", int(args.test))
            mlflow.log_metric("Time", total_time.seconds)


def main(args: argparse.Namespace) -> int:
    """Main execution function.

    This function loads data, applies the language model pipeline,
    logs performance, and saves output notes.
    """

    if args.scope not in {Constants.FUTURE_VISION, Constants.STRATEGY_INPUT}:
        logger.error("Unsupported scope: %s", args.scope)
        return 1

    if sys.platform == 'win32':
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("app_run_v2")
        with mlflow.start_run(run_name="trend_week_report_step3") as run:
            run_id = run.info.run_id
    else:
        mlflow.set_tracking_uri(uri=os.environ['MLFLOW_URL'])
        mlflow.set_experiment(os.environ['WORKFLOWNAME'])
        run_id = load_run_id(args)

    model = model_activation(args.model_name)
    prompt_template, output_parser = load_prompt_template(args)

    # Check if the module exists
    note_pipeline = RunnablePassthrough.assign(result=prompt_template | model | output_parser)

    input_path = resolve_report_raw_path(args, ext=Constants.CSV)

    df = pd.read_csv(input_path)

    batch = build_batches(df)

    begin = datetime.now()
    notes = note_pipeline.batch(batch)
    end = datetime.now()
    total_time = end - begin

    logger.info(f"{args.scope}_report_step2 total time: {total_time}")

    results = {note['filename']: note['result'].model_dump() for note in notes}

    save_results(results, args, ext=Constants.JSON)

    log_mlflow_metrics(run_id, args, total_time)

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Flag to enable debugging')
    parser.add_argument('--scope', type=str, help='project')
    parser.add_argument('--source', type=str, required=False,
                        help='source of data. necessary for project `strategy_input`')
    parser.add_argument("--model_name", type=str, default="gpt-4.1-2025-04-14")
    args = parser.parse_args()

    directory = os.path.join(
        get_datafetch(),
        *([Constants.REPORT] if args.scope == Constants.FUTURE_VISION
                                          else [args.source, Constants.REPORT])
    )

    if not os.path.isdir(directory):
        logger.info(f"Creating output directory at {directory}")
        os.makedirs(directory)

    main(args)