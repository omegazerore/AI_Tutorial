import io
import base64
from typing import List

import pandas as pd
from PIL import Image
from langchain_core.runnables import Runnable


def image_to_base64(image_path):
    """Converts an image to a Base64-encoded JPEG string.

    Args:
        image_path: Path to the image file or a file-like object.

    Returns:
        Base64-encoded string of the JPEG image.
    """
    with Image.open(image_path) as image:
        # Save the Image to a Buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

        # Encode the Image to Base64
        image_str = base64.b64encode(buffered.getvalue())

    return image_str.decode('utf-8')


def save_results_to_csv(data, path: str, columns: List[str]) -> None:
    """Saves structured data to a CSV file.

    Args:
        data: Either a list of dictionaries or a list of lists representing rows.
        path: Output CSV file path.
        columns: List of column names for the CSV.

    Raises:
        ValueError: If the data format is not supported.
    """
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # Case: list of dictionaries
        df = pd.DataFrame(data)
        df = df[columns]
    elif isinstance(data, list):
        # Case: list of lists
        df = pd.DataFrame(data, columns=columns)
    else:
        raise ValueError("Unsupported data format")

    df.to_csv(path, index=False)


def build_pipeline(steps: list) -> Runnable:
    """Builds a sequential pipeline by chaining Runnable steps.

    Args:
        steps: A list of Runnable objects to be executed in order.

    Returns:
        A single Runnable representing the composed pipeline.

    Raises:
        ValueError: If the steps list is empty.
    """
    if not steps:
        raise ValueError("Steps list cannot be empty.")

    pipeline = steps[0]
    for step in steps[1:]:
        pipeline = pipeline | step
    return pipeline


STEP_1_FILENAME = "Fashion_Product_Release.csv"
STEP_2_TEXT_FILENAME = "text.csv"
STEP_2_IMAGE_FILENAME = "image_signature.csv"
STEP_3_FILENAME = "product_news.csv"
MAX_CONCURRENCY = 10
