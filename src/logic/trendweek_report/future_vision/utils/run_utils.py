import argparse
import os

from src.io.path_definition import get_datafetch
from src.logic.trend_week_report.future_vision.constants import Constants


def load_run_id(args: argparse.Namespace) -> str:
    """Retrieves the MLflow run ID from a stored text file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: The run ID associated with the given project scope and source.
    """
    if args.scope == Constants.FUTURE_VISION:
        run_path = os.path.join(get_datafetch(), f"{args.scope}_run_id.txt")
    else:
        run_path = os.path.join(get_datafetch(), f"{args.scope}_{args.source}_run_id.txt")
    with open(run_path, "r") as f:
        return f.read().strip()