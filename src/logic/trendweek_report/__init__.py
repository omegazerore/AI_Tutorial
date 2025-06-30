import os
import logging
from typing import Optional

from src.io.path_definition import get_datafetch


def setup_logger(scope: str, source: Optional[str]=None):

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 1. File handler
    if scope in ['future_vision']:
        file_handler = logging.FileHandler(os.path.join(get_datafetch(), f"report_step1.log"))
    else:
        file_handler = logging.FileHandler(os.path.join(get_datafetch(), source, f"report_step1.log"))
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger