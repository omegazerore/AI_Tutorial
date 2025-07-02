import argparse
import os

from src.io.path_definition import get_datafetch
from src.logic.trend_week_report.future_vision.constants import Constants


def resolve_report_raw_path(args: argparse.Namespace, ext: str) -> str:
    """Generates the file path for the raw report output.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        ext (Constants): Desired file extension (e.g., Constants.CSV, Constants.JSON).

    Returns:
        str: Full path to the report file.
    """
    filename = f"report_raw{'_test' if args.test else ''}.{ext}"
    if args.scope == Constants.FUTURE_VISION:
        return os.path.join(get_datafetch(), filename)
    return os.path.join(get_datafetch(), args.source, filename)