"""
PDF Processing Pipeline

This script processes PDF files by separating them into individual pages,
extracting text and image elements using the `unstructured` library, moving
associated images, and compiling the results into a CSV file.

The main processing steps include:
- Splitting multi-page PDFs into single-page PDFs.
- Extracting elements (text and images) from each page.
- Organizing and moving image files associated with PDF pages.
- Aggregating extracted data and saving it to a CSV.

Requires:
    - nltk
    - PyPDF2
    - unstructured
    - tqdm

Example:
    Run the script from the command line:
        python script.py --scope projectA --source docset1
"""

import os
import shutil
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from PyPDF2 import PdfWriter, PdfReader
from PyPDF2.errors import PdfReadError
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element

from src.logic.product_news import save_results_to_csv
from src.io.path_definition import get_datafetch
from src.logic.product_news import STEP_1_FILENAME


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Constants / Configuration ===
FIGURE_DIR_NAME = "figure"
CHUNKING_STRATEGY = "by_title"
IMAGE_BLOCK_TYPES = ["Image"]
MAX_CHARACTERS = 4000
NEW_AFTER_N_CHARS = 3800
COMBINE_TEXT_UNDER_N_CHARS = 2000
OVERLAP = 200
PDF_SPLIT_NAME_TEMPLATE = "{base}-page-{page:02d}.pdf"


# === Function Definitions ===
def pdf_file_separation(filename: str) -> List[str]:
    """Splits a PDF file into single-page PDF files.

    Args:
        filename (str): The path to the input multi-page PDF.

    Returns:
        List[str]: A list of file paths to the single-page PDF files.
    """
    input_path = Path(filename)
    output_dir = input_path.parent
    base = input_path.stem

    output_filenames = []

    with open(filename, "rb") as file:
        inputpdf = PdfReader(file)

        for i, page in enumerate(inputpdf.pages):
            output = PdfWriter()
            output.add_page(page)

            new_file_name = os.path.join(output_dir, PDF_SPLIT_NAME_TEMPLATE.format(base=base, page=i))
            output_filenames.append(new_file_name)

            with open(new_file_name, "wb") as outputStream:
                output.write(outputStream)

    return output_filenames


def extract_element(filename: str, image_dir: str) -> List[Element]:
    """Extracts text and image elements from a PDF page using `unstructured`.

    Args:
        filename (str): The path to the single-page PDF.
        image_dir (str): Directory to save extracted images.

    Returns:
        List[Element]: A list of extracted document elements.
    """
    elements = partition_pdf(filename,
                             chunking_strategy=CHUNKING_STRATEGY,
                             infer_table_structure=True,
                             extract_image_block_types=IMAGE_BLOCK_TYPES,
                             max_characters=MAX_CHARACTERS,
                             new_after_n_chars=NEW_AFTER_N_CHARS,
                             combine_text_under_n_chars=COMBINE_TEXT_UNDER_N_CHARS,
                             overlap=OVERLAP,
                             extract_image_block_output_dir=image_dir,
                             strategy='hi_res',
                             multipage_sections=False)

    logger.debug(f"Partitioned {len(elements)} elements from file {filename}")

    return elements

def move_images(page_number: int, source_dir: str, target_dir: str) -> None:
    """Renames and moves extracted image files from a temporary directory to a target location.

    Args:
        page_number (int): The page number used to rename images.
        source_dir (str): Directory containing extracted images.
        target_dir (str): Destination directory for moved images.
    """
    if os.path.exists(source_dir):
        logger.debug(f"Image files found: {os.listdir(source_dir)}")
        for image_file in os.listdir(source_dir):
            image_file_list = image_file.split("-")
            image_file_list[1] = str(page_number)

            img_src = os.path.join(source_dir, image_file)
            img_dst = os.path.join(target_dir, "-".join(image_file_list))

            logger.debug(f"Moving image from {img_src} to {img_dst}")
            shutil.move(img_src, img_dst)

        shutil.rmtree(source_dir)
        logger.debug(f"Removed temporary image directory: {source_dir}")


def process_pdf(filename: str, page_number: int, output_dir: str, image_dir: str) -> List[List[str]]:
    """Processes a single-page PDF to extract data and move associated images.

    Args:
        filename (str): Path to the single-page PDF.
        page_number (int): Page number of the PDF being processed.
        output_dir (str): Base output directory for temporary image files.
        image_dir (str): Final directory for storing moved images.

    Returns:
        List[List[str]]: Extracted data rows, each containing text, filename, and page number.
    """
    data = []
    sep_image_dir = os.path.join(output_dir, FIGURE_DIR_NAME, Path(filename).stem)

    logger.debug(f"Processing PDF: {filename}, page {page_number}")
    logger.debug(f"Temporary image directory: {sep_image_dir}")

    try:
        elements = extract_element(filename=filename, image_dir=sep_image_dir)

        if os.path.exists(sep_image_dir):
            move_images(page_number=page_number, source_dir=sep_image_dir, target_dir=image_dir)

        for element in elements:
            logger.debug(f"Extracted text: {element.text[:100]}...")
            data.append([element.text, Path(filename).stem, page_number])
    finally:
        if os.path.exists(filename):
            os.remove(filename)
            logger.debug(f"Deleted single-page PDF: {filename}")

    return data


def main(scope: str, source: str) -> None:
    """Main entry point for PDF processing.

    Processes all PDFs in the given directory, splits them into pages,
    extracts data and images, and stores the results in a CSV file.

    Args:
        scope (str): The scope or project directory under the data fetch base path.
        source (str): The specific data source or subdirectory within the scope.
    """
    data = []

    dir_ = os.path.join(get_datafetch(), scope, source, 'report')
    logger.debug(f"Fetching PDFs from directory: {dir_}")

    if not os.path.isdir(dir_):
        os.makedirs(dir_)
        sys.exit(f"Please upload files in {dir_}")

    for filename in tqdm(os.listdir(dir_)):
        if filename.endswith('.pdf'):
            logger.info(f"Processing file: {filename}")

            image_dir = os.path.join(dir_, FIGURE_DIR_NAME, Path(filename).stem)

            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
                logger.debug(f"Created image directory: {image_dir}")

            try:
                sep_filenames = pdf_file_separation(os.path.join(dir_, filename))
                logger.debug(f"Split PDF into {len(sep_filenames)} pages")
            except FileNotFoundError:
                logger.error(f"File not found: {filename}. Skipping.")
                continue
            except PdfReadError as e:
                logger.error(f"PyPDF2 failed to read {filename}: {e}. Skipping.")
                continue
            except PermissionError:
                logger.error(f"Permission denied when accessing {filename}. Skipping.")
                continue
            except OSError as e:
                logger.error(f"OS-level error with file {filename}: {e}. Skipping.")
                continue

            for page_number, sep_filename in enumerate(sep_filenames):
                logger.debug(f"Processing page {page_number} of {filename}")
                data_page = process_pdf(filename=sep_filename, page_number=page_number,
                                        output_dir=dir_, image_dir=image_dir)

                data.extend(data_page)

    filename = os.path.join(get_datafetch(), scope, source, STEP_1_FILENAME)
    logger.info(f"Saving extracted data to {filename}")
    save_results_to_csv(data=data, path=filename, columns=['text', 'file', 'page_number'])


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    main(args.scope, args.source)