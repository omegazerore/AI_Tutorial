"""
Pass test 2025.06. - Meng-Chieh Ling
"""
import argparse
import os
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
from openai import OpenAI

from src.initialization import credential_init
from src.io.path_definition import get_datafetch
from src.logic.product_news.websearch_service import WebSearchService
from src.logic.product_news.signature_extraction import SignatureExtraction, Signature2Brand
from src.logic.product_news import MAX_CONCURRENCY, STEP_2_TEXT_FILENAME, STEP_2_IMAGE_FILENAME, save_results_to_csv


# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def fetch_path(*parts) -> str:
    """Constructs a path by joining the base fetch directory with additional parts.

    Args:
        *parts: Variable length path components.

    Returns:
        Full file path as a string.
    """
    return os.path.join(get_datafetch(), *parts)


def image_2_brand(image_dir: str, legit_page_number: List[str],
                  signature_extraction_pipeline,
                  signature2brand_pipeline) -> List[Dict]:
    """Extracts brand information from images using signature extraction and conversion pipelines.

    Args:
        image_dir: Directory containing image files.
        legit_page_number: List of valid page numbers for filtering images.
        signature_extraction_pipeline: Pipeline to extract signature from images.
        signature2brand_pipeline: Pipeline to convert signature to brand.

    Returns:
        A list of dictionaries containing brand information for each image.
    """
    logging.info(f"Scanning directory: {image_dir}")
    batch_input = []

    try:
        image_files = os.listdir(image_dir)
    except FileNotFoundError:
        logging.error(f"Directory not found: {image_dir}")
        return []
    except PermissionError:
        logging.error(f"Permission denied accessing directory: {image_dir}")
        return []
    except Exception as e:
        logging.exception(f"Unexpected error while accessing directory: {e}")
        return []

    for image_file in image_files:
        try:
            page_number = image_file.split("-")[1]
        except IndexError:
            logging.warning(f"Unexpected image filename format: {image_file}")
            continue  # or log and skip malformed filenames

        if page_number in legit_page_number:
            # if image_file in ["figure-4-5.jpg", "figure-6-1.jpg", "figure-8-4.jpg"]:
            batch_input.append({"image_path": os.path.join(image_dir, image_file),
                                "page_number": int(page_number),
                                # "file": Path(image_dir).stem,
                                "image_file_name": image_file})

    if not batch_input:
        logging.warning("No valid images to process.")
        return []

    logging.info(f"Extracting signatures from {len(batch_input)} images")

    try:
        answers = signature_extraction_pipeline.batch(batch_input, max_concurrency=MAX_CONCURRENCY)
    except Exception as e:
        logging.exception("Failed during signature extraction batch processing.")
        return []

    df = pd.DataFrame(data=answers)

    required_columns = {'page_number', 'image_file_name', 'signature'}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        logging.error(f"Missing required columns for processing: {missing_columns}")
        return []

    # Safely select only existing columns
    df = df[list(required_columns)]
    df = df[df['signature'] != ""]

    if df.empty:
        logging.warning("Filtered DataFrame is empty after removing rows without valid signatures.")
        return []

    batch_output = df.to_dict(orient='records') #[row.to_dict() for _, row in df.iterrows()]

    logging.info("Converting signatures to brands")
    try:
        output = signature2brand_pipeline.batch(batch_output, max_concurrency=MAX_CONCURRENCY)
    except Exception as e:
        logging.exception("Failed during signature-to-brand batch processing.")
        return []

    return output


def main(scope: str, source: str, model_name: str, search_context_size: str,
         websearch_model: str):
    """Main processing function.

    Reads strategy data from a CSV, processes associated images to extract brand info,
    and writes the results to a new CSV.

    Args:
        scope: Directory scope (e.g., project or data type).
        source: Source identifier or folder.
        model_name: Name of the model used for extraction and conversion.
        search_context_size: Context size for web search queries.
        websearch_model: Model name used by the web search service.
    """
    logging.info("Starting main process")
    credential_init()
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    web_search_service = WebSearchService(client, search_context_size=search_context_size,
                                          model=websearch_model)

    signature_extraction_pipeline = SignatureExtraction(model_name="gpt-4.1").pipeline
    signature2brand_pipeline = Signature2Brand(model_name=model_name, web_search_service=web_search_service)

    csv_filename = fetch_path(scope, source, STEP_2_TEXT_FILENAME)
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_filename}")
        return
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file {csv_filename}: {e}")
        return

    required_columns = {'file', 'page_number', 'if_strategy'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

    df = df[df['if_strategy'] == True]
    if df.empty:
        logging.warning("No rows with 'if_strategy' == True in the CSV.")
        return

    data = []

    # df = df[df['file']=='Beauty_Textures_Forecast_2027_en.pdf']

    for file in df['file'].unique():
        image_dir = fetch_path(scope, source, 'figure', Path(file).stem)

        legit_page_number = [str(pn) for pn in df[df['file'] == file]['page_number'].to_list()]

        logging.info(f"Processing file: {file}")
        sub_data = image_2_brand(image_dir=image_dir, legit_page_number=legit_page_number,
                                  signature_extraction_pipeline=signature_extraction_pipeline,
                                  signature2brand_pipeline=signature2brand_pipeline)
        for a in sub_data:
            a['file'] = file
        data.extend(sub_data)

    if not data:
        logging.warning("No data to write. Skipping file save.")
        return

    csv_filename = fetch_path(scope, source, STEP_2_IMAGE_FILENAME)

    logging.info(f"Saving results to {csv_filename}")
    save_results_to_csv(data=data, path=csv_filename,
                        columns=['page_number', 'file', 'image_file_name', 'signature', 'brand', 'websearch_text', 'country_code'])

    logging.info("Process completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', type=str, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--search_context_size', type=str, required=True)
    parser.add_argument('--websearch_model', type=str, required=True)
    args = parser.parse_args()

    main(scope=args.scope, source=args.source, model_name=args.model_name,
         search_context_size=args.search_context_size, websearch_model=args.websearch_model)

