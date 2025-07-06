import argparse
import importlib
import importlib.util
import logging
import os
import sys
from datetime import datetime

import mlflow

from src.logic.trend_week_report.constants import Constants
from src.logic.trend_week_report.utils.paths import resolve_report_raw_path
from src.logic.trend_week_report.utils.run_utils import load_run_id


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_mlflow_metrics(run_id: str, args: argparse.Namespace, total_time):
    """Logs MLflow parameters and execution metrics.

    Args:
        run_id (str): Existing MLflow run ID to log into.
        args (argparse.Namespace): Parsed command-line arguments.
        total_time (timedelta): Duration of the report generation process.
    """
    with mlflow.start_run(run_id=run_id):
        run_name = f"{args.scope}_report_step3" if args.scope == Constants.FUTURE_VISION else f"{args.scope}_{args.source}_report_step3"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_param("LLM Engine", args.model_name)
            mlflow.log_param("test", int(args.test))
            mlflow.log_metric("Time", total_time.seconds)


def load_report_generator(args: argparse.Namespace):
    """Dynamically loads and returns the appropriate report generator module.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        object: An instance of the TrendWeekReport class from the loaded module.
    """
    module_name = f"src.logic.trend_week_report.{args.scope}.report_pipeline"
    spec = importlib.util.find_spec(module_name)
    if not spec:
        logger.error(f"Module '{module_name}' not found.")
        sys.exit(1)
    module = importlib.import_module(module_name)

    report_json_file = resolve_report_raw_path(args, ext=Constants.JSON)

    return module.TrendWeekReport(report_json_file=report_json_file,
                                  model=args.model_name)


def main(args) -> int:
    """Main execution logic for generating and logging a report.

    Handles tracking setup, report generation, metric logging, and file saving.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        int: Exit code (0 for success, 1 for error).
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

    begin = datetime.now()

    report_generator = load_report_generator(args)

    if args.scope == Constants.FUTURE_VISION:
        final_doc = report_generator.run(search_context_size=args.search_context_size,
                                         search_model=args.search_model_name)
    if args.scope == Constants.STRATEGY_INPUT:
        final_doc = report_generator.run()

    end = datetime.now()
    total_time = end - begin

    log_mlflow_metrics(run_id, args, total_time)

    output_filename = resolve_report_raw_path(args, ext=Constants.WORD)

    final_doc.save(output_filename)

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pipeline for project: future_vision or strategy_input'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Enable debugging mode.'
    )
    parser.add_argument(
        '--scope',
        type=str,
        required=True,
        choices=['future_vision', 'strategy_input'],
        help='Project scope. Allowed: future_vision or strategy_input.'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=False,
        help=('Source of data. Required for scope "strategy_input". '
              'Options: Beautystreams, ForesightFactory, globalData, Stylus, wgsn, OtherSource.')
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt-4.1-2025-04-14',
        help='Model name (default: gpt-4.1-2025-04-14)'
    )
    parser.add_argument(
        '--search_context_size',
        type=str,
        required=False,
        help='OpenAI websearch search_context_size. Required if scope is future_vision.'
    )
    parser.add_argument(
        '--search_model_name',
        type=str,
        required=False,
        help='OpenAI websearch model. Required if scope is future_vision.'
    )
    args = parser.parse_args()

    # Custom validation
    if args.scope == Constants.FUTURE_VISION:
        if not args.search_context_size:
            parser.error(f'--search_context_size is required when --scope is {Constants.FUTURE_VISION}')
        if not args.search_model_name:
            parser.error(f'--search_model_name is required when --scope is {Constants.FUTURE_VISION}')
    if args.scope == Constants.STRATEGY_INPUT and not args.source:
        parser.error(f'--source is required when --scope is {Constants.STRATEGY_INPUT}')

    return args



if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--test', action='store_true', help='Flag to enable debugging')
    # parser.add_argument('--scope', type=str, help='project future_vision or strategy_input')
    # parser.add_argument('--source', type=str, required=False,
    #                     help='source of data. necessary for project `strategy_input`. Now we have Beautystreams, ForesightFactory, globalData, Stylus, wgsn, OtherSource')
    # parser.add_argument("--model_name", type=str, default="gpt-4.1-2025-04-14")
    # parser.add_argument("--search_context_size", type=str, help="OpenAI websearch search_context_size")
    # parser.add_argument("--search_model_name", type=str, help="OpenAI websearch model")
    # args = parser.parse_args()

    args = parse_args()

    main(args)