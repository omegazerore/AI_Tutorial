"""Module for extracting structured cosmetic strategy and product concepts from PDF reports.

This script uses LangChain for LLM-based text processing, and pymupdf for PDF parsing.
Pass test 2025.06.23 - Meng-Chieh Ling
"""

import os
import logging
from pathlib import Path
from typing import List, Dict

import pymupdf
import pandas as pd
from textwrap import dedent
from tqdm import tqdm
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, chain, Runnable
from pydantic import BaseModel, Field

from src.initialization import model_activation
from src.io.path_definition import get_datafetch
from src.logic.trend_week_report import build_standard_chat_prompt_template
from src.logic.product_news import STEP_1_FILENAME, MAX_CONCURRENCY, STEP_2_TEXT_FILENAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FONT = 'DMSans-Bold'
SIZE = 15
FILE = "file"
PAGE_NUMBER = "page_number"


class Product(BaseModel):
    """Model for a single product name."""
    name: str = Field(description="Product")
    brand: str = Field(description="The brand name")
    country_code: str = Field(description="ISO 3166-1 alpha-2 of the country of the brand")


class ProductOutput(BaseModel):
    """Container model for a list of products."""
    products: List[Product] = Field(description="A list of products")


class StrategyOutput(BaseModel):
    """Model representing a strategy relevance judgment."""
    name: bool = Field(description="If this text is related to skin care or color cosmetic")
    reason: str = Field(description="why do you think this is related to skin care or color cosmetic")


def _build_product_parser():
    """Builds a parser for extracting products using Pydantic."""
    output_parser = PydanticOutputParser(pydantic_object=ProductOutput)
    format_instructions = output_parser.get_format_instructions()

    return output_parser, format_instructions


def _build_strategy_parser():
    """Builds a parser for determining relevance to strategy."""
    output_parser = PydanticOutputParser(pydantic_object=StrategyOutput)
    format_instructions = output_parser.get_format_instructions()

    return output_parser, format_instructions


def build_product_extraction_prompt_template(format_instructions):
    """Creates a product extraction prompt chain.

    Args:
        format_instructions: Instructions on how to format the output.

    Returns:
        A LangChain Runnable prompt template.
    """
    system_template = dedent("""
        You are a helpful and detail-oriented AI assistant working as the personal assistant 
        to a trend manager in the beauty and skincare industry.

        Your primary task is to carefully read and analyze provided text and accurately 
        extract any cosmetic or skincare products mentioned.

        You always strive for precision, pay close attention to details, and ensure that 
        no relevant products are missed.

        Only include products or product concepts that are clearly associated with a specific brand, creator, or company.
        These can include branded product names, as well as emerging or conceptual product formats.

        Do not include general product types or formats unless they are linked to a brand or identifiable source.

        Output should be a clean, concise list of product or product concept references, each clearly connected to a brand or creator.
    """)

    human_template = dedent("""
        {text}

        Output format instructions: {format_instructions}
    """)

    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                        "input_variables": ['text'],
                        "partial_variables": {"format_instructions": format_instructions}}
              }

    return build_standard_chat_prompt_template(input_)


def build_strategy_extraction_prompt_template(format_instructions):
    """Creates a strategy relevance extraction prompt chain.

    Args:
        format_instructions: Instructions on how to format the output.

    Returns:
        A LangChain Runnable prompt template.
    """

    system_template = dedent("""
            You are a highly knowledgeable, analytical, and detail-oriented AI assistant 
            serving as a personal assistant to a trend manager in the beauty and skincare industry.

            Your main task is to evaluate whether a given 'concept' refers to or is relevant 
            to a skincare or color cosmetic product.

            The concept has been automatically extracted from a PDF document and may not represent a real product idea. 
            It could be unrelated text, such as a page number, section heading, or label.

            You are provided with additional contextual text from the same PDF page to aid your judgment.

            Carefully analyze the concept together with its context.

            If the concept does not clearly relate to a skincare or color cosmetic product or idea, 
            it should be classified as *not relevant*. Do not make assumptions based on weak or vague connections.

            Err on the side of caution: when in doubt, treat the concept as unrelated to cosmetics or skincare.

            Your assessment must be accurate, well-reasoned, and consistent.
        """)

    human_template = dedent("""
            Concept: {strategy}

            Context:

            {text}

            Please follow this output format: {format_instructions}
        """)

    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                        "input_variables": ['strategy', 'text'],
                        "partial_variables": {"format_instructions": format_instructions}}
              }

    return build_standard_chat_prompt_template(input_)


class TextExtraction:
    """LLM-powered text extraction pipeline for products and strategy relevance."""

    def __init__(self, product_prompt_template: Runnable, strategy_prompt_template: Runnable, model_name: str):

        model = model_activation(model_name=model_name)

        product_output_parser, _ = _build_product_parser()
        strategy_output_parser, _ = _build_strategy_parser()


        self.pipeline_ = RunnablePassthrough.assign(products=product_prompt_template | model | product_output_parser | self.product_extraction,
                                                    if_strategy=strategy_prompt_template | model | strategy_output_parser | self.extract_strategy_flag)

    @chain
    @staticmethod
    def product_extraction(kwargs):
        """Extracts product names from Pydantic output."""
        return kwargs.model_dump()#[product.name for product in kwargs.products]

    @chain
    @staticmethod
    def extract_strategy_flag(kwargs):
        """Extracts the strategy relevance flag."""
        return kwargs.name


def structure_extraction_subroutine(filename: str) -> List[Dict]:
    """Extracts concept candidates from a PDF.

    Args:
        filename: Path to the PDF file.

    Returns:
        A list of extracted text snippets (concepts) with metadata.
    """
    concept = []

    doc = pymupdf.open(filename)
    for idx, page in enumerate(doc):
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            if block['type'] != 0:
                continue
            lines = block['lines']
            for line in lines:
                spans = line['spans']
                for span in spans:
                    font = span['font']
                    size = span['size']
                    if font == FONT and size == SIZE:
                        concept.append({PAGE_NUMBER: idx,
                                        "strategy": span['text'],
                                        FILE: Path(filename).name})

    return concept


def structure_extraction(scope: str, source: str) -> pd.DataFrame:
    """Runs concept extraction over all PDFs in a target directory.

    Args:
        scope: Project scope folder.
        source: Source dataset or project ID.

    Returns:
        DataFrame with extracted concepts.
    """
    dir_ = os.path.join(get_datafetch(), scope, source, 'report')
    logger.debug(f"Fetching PDFs from directory: {dir_}")

    if not os.path.isdir(dir_):
        os.makedirs(dir_)
        raise FileNotFoundError(f"Expected PDFs in {dir_}, but directory was empty or missing.")

    concept = []

    for filename in tqdm(f for f in os.listdir(dir_) if f.endswith('.pdf')):
        logger.info(f"Processing file: {filename}")
        try:
            concept.extend(structure_extraction_subroutine(os.path.join(dir_, filename)))
        except Exception as e:
            logger.error(f"Failed to open {filename}: {e}")
            continue

    return pd.DataFrame(data=concept)


def main(scope: str, source: str, model_name: str):
    """Main entry point for running product and strategy extraction.

    Args:
        scope: Project scope.
        source: Source project ID.
    """
    logger.info(f"Started text extraction for scope='{scope}', source='{source}'.")

    # Step 1: Extract concepts from PDFs
    logger.info("Starting concept extraction from PDF files.")
    concept =  structure_extraction(scope=scope, source=source)
    logger.debug(f"Extracted {len(concept)} concept entries.")

    # Step 2: Build prompt templates
    logger.info("Building product and strategy prompt templates.")
    _, product_format_instructions = _build_product_parser()
    product_prompt_template = build_product_extraction_prompt_template(product_format_instructions)

    _, strategy_format_instructions = _build_strategy_parser()
    strategy_prompt_template = build_strategy_extraction_prompt_template(strategy_format_instructions)

    # Step 3: Initialize the text extraction pipeline
    logger.info("Initializing the TextExtraction pipeline.")
    text_extraction = TextExtraction(product_prompt_template=product_prompt_template,
                                     strategy_prompt_template=strategy_prompt_template,
                                     model_name=model_name)

    # Step 4: Load extracted text CSV
    filename = os.path.join(get_datafetch(), scope, source, STEP_1_FILENAME)
    logger.info(f"Loading input CSV from: {filename}")
    try:
        df = pd.read_csv(filename)
        logger.debug(f"Loaded dataframe with shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {filename}")
        raise

    # Step 5: Merge with concept metadata
    logger.info("Merging extracted concepts with input CSV.")
    # concept = concept[concept['strategy']=='Gel-powders']
    df = df.merge(concept, on=[PAGE_NUMBER, FILE])
    if df.empty:
        logger.warning("Merged dataframe is empty — check if concepts matched any page/file entries.")
    else:
        logger.debug(f"Merged dataframe shape: {df.shape}")

    # Step 6: Prepare batch input for LLM
    batch = df.to_dict(orient='records')
    logger.info(f"Preparing {len(batch)} rows for LLM processing.")

    # Step 7: Run the extraction pipeline
    logger.info("Running batch inference via TextExtraction pipeline.")
    output = text_extraction.pipeline_.batch(batch, config={"max_concurrency": MAX_CONCURRENCY})
    logger.debug("Inference completed successfully.")

    # Step 8: Save output
    final_output = []
    for row in output:
        products = row['products']['products']
        if len(products) > 0:
            for product in products:
                appended = row.copy()
                del appended['products']
                appended.update(product)
                final_output.append(appended)

    output_df = pd.DataFrame(final_output)

    csv_filename = os.path.join(get_datafetch(), scope, source, STEP_2_TEXT_FILENAME)
    output_df.to_csv(csv_filename, index=False)
    logger.info(f"Extraction results saved to: {csv_filename}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', type=str, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    main(scope=args.scope, source=args.source, model_name=args.model_name)

