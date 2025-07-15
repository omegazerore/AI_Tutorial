"""
EVY Technology Daily Defence Face Mousse Sonnencreme online kaufen
EVY TECHNOLOGY UV / Heat Hair Mousse Sonnenspray ✔️ online kaufen | DOUGLAS
[TIRTIR] Off The Sun Air Mousse SPF 50+ PA++++ 100ml - COCOMO
BREAD BEAUTY SUPPLY - Hair-Foam: Curling Mousse | Ulta Beauty
NUSE Mousse Liptual – Ma Petite Coree
The Dream Collection Whipped Shower Foam - Reichhaltiger Duschschaum | RITUALS
Mousse de Banho Baunilha | Espuma hidratante para um banho perfumado
"""
import argparse
import logging
import os
from functools import partial
from typing import List, Tuple

import pandas as pd
from openai import OpenAI
from textwrap import dedent
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.initialization import model_activation, credential_init
from src.io.path_definition import get_datafetch
from src.logic.product_news.websearch_service import WebSearchService
from src.logic.trend_week_report import build_standard_chat_prompt_template
from src.logic.product_news import STEP_2_TEXT_FILENAME, STEP_2_IMAGE_FILENAME, STEP_3_FILENAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


REQUIRED_TEXT_COLUMNS = ['name', 'if_strategy', 'page_number', 'file', 'brand', 'country_code']
COUNTRY_CODE_MAP = {"UK": "GB"}


class Source(BaseModel):
    """Schema for individual product source information."""

    content: str = Field(description="description of the product.")
    url: str = Field(description="url of the content source.")
    name: str = Field(description="The product, including the brand name.")
    brand: str = Field(description="the brand name")


class Output(BaseModel):
    """Schema for parsed output data containing product sources."""

    name: List[Source] = Field(description="A list of content and url")


output_parser = PydanticOutputParser(pydantic_object=Output)
format_instructions = output_parser.get_format_instructions()


def safe_read_csv(path: str) -> pd.DataFrame:
    """Safely reads a CSV file into a DataFrame.

    Args:
        path: The path to the CSV file.

    Returns:
        A pandas DataFrame if successful, otherwise an empty DataFrame.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"[ERROR] File not found: {path}")
    except pd.errors.EmptyDataError:
        logger.error(f"[ERROR] File is empty: {path}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to read {path}: {e}")
    return pd.DataFrame()


def websearch_text(kwargs, websearch_engine: WebSearchService):
    """Performs text-based product web search.

    Args:
        kwargs: Dictionary containing product context including strategy, name, brand, and country_code.
        websearch_engine: An instance of WebSearchService for querying.

    Returns:
        Web search result.
    """

    strategy = kwargs['strategy']
    product = kwargs['name']
    brand = kwargs['brand']
    country_code = kwargs['country_code']
    country_code = COUNTRY_CODE_MAP.get(country_code, country_code)

    messages = [{"role": "user",
                 "content": dedent(f"""
                 What is the cosmetic/skincare product {brand} {product} under the concept {strategy}?
                 Please find the best match and the corresponding product page, in which one can order the product.
                 Return only one product page.
                 """)}]

    result = websearch_engine.search(messages, country_code=country_code)

    return result


def websearch_image(kwargs, websearch_engine: WebSearchService):
    """Performs image-contextual product web search.

    Args:
        kwargs: Dictionary containing image-related context including strategy, text, brand, and country_code.
        websearch_engine: An instance of WebSearchService for querying.

    Returns:
        Web search result.
    """

    strategy = kwargs['strategy']
    context = kwargs['text']
    brand = kwargs['brand']
    country_code = kwargs['country_code']
    if pd.isnull(kwargs['country_code']):
        country_code = None
    country_code = COUNTRY_CODE_MAP.get(country_code, country_code)

    messages = [{"role": "user",
                 "content": f"What is the cosmetic or skin care product under the concept {strategy} with brand {brand}?\n\n"
                            f"The concept has the following context: {context}\n\n"
                            f"Please provide me the page to the product of the brand.\n"
                            f"If the exact product cannot be found, please give me products with similar concept within {brand}.\n"
                            f"Ideally from the official website of the brand {brand}."
                 }]

    result = websearch_engine.search(messages, country_code=country_code)

    return result


def build_extraction_prompt():
    """Builds a prompt template for extracting product content and URLs.

    Returns:
        A runnable prompt template for the extraction task.
    """

    system_template = (
        "You are a helpful assistant. Your task is to extract all relevant product descriptions "
        "and their corresponding source URLs from a given input text.\n\n"
        "Each extracted entry should contain:\n"
        "- `content`: A concise and coherent description of a product or item.\n"
        "- `url`: The URL associated with that content, pointing to the source.\n\n"
        "Ensure the output strictly follows the specified structure using the provided format instructions. "
        "Do not include unrelated information or URLs without associated content. "
        "If no valid pairs are found, return an empty list."
    )

    human_template = "text: {text}\noutput format instruction: {format_instructions}"

    input_ = {"system": {"template": system_template,
                         },
              "human": {"template": human_template,
                        "input_variables": ['text'],
                        "partial_variables": {"format_instructions": format_instructions}}}

    prompt_template = build_standard_chat_prompt_template(input_)

    return prompt_template


def parse_pipeline_output(output, source: str) -> List:
    """Parses pipeline output and attaches source metadata.

    Args:
        output: List of results from pipeline.
        source: Label indicating the source type ('text' or 'image').

    Returns:
        A flat list of parsed product data entries.
    """

    parsed_output = []

    for c in output:
        results = c['results'].name
        for result in results:
            c_copy = {k: v for k, v in c.items() if k != 'results'}
            c_copy.update(result.model_dump())
            c_copy["source_data_type"] = source
            parsed_output.append(c_copy)

    return parsed_output


def prepare_image_data(df_image: pd.DataFrame, df_text: pd.DataFrame) -> pd.DataFrame:
    """Prepares image DataFrame by merging and cleaning.

    Args:
        df_image: DataFrame with image data.
        df_text: DataFrame with textual context data.

    Returns:
        A merged DataFrame with contextual fields.
    """

    df_metadata = df_text.drop_duplicates(subset=['page_number', 'file'])[['page_number', 'file', 'strategy', 'text']]

    df_image = df_image.merge(df_metadata, on=['page_number', 'file'])
    # df_image = df_image[df_image['image_file_name']=='figure-10-3.jpg']
    df_image.dropna(subset=['brand'], inplace=True)

    return df_image


def run_pipeline(df_text: pd.DataFrame, df_image: pd.DataFrame, text_pipeline: Runnable, image_pipeline: Runnable) -> pd.DataFrame:
    """Runs the full data processing pipeline for text and image data.

    Args:
        df_text: DataFrame with text-based input records.
        df_image: DataFrame with image-based input records.
        text_pipeline: Runnable text pipeline.
        image_pipeline: Runnable image pipeline.

    Returns:
        A DataFrame with combined parsed output from both pipelines.
    """
    output = []

    try:
        batch_text = df_text[df_text['if_strategy']].to_dict(orient="records")
        output_text = text_pipeline.batch(batch_text)
        output.extend(parse_pipeline_output(output_text, 'text'))
    except Exception as e:
        logger.exception("[ERROR] Failed in text pipeline: %s", e)

    try:
        df_image = prepare_image_data(df_image, df_text)
        batch_image = df_image.to_dict(orient="records")
        output_image = image_pipeline.batch(batch_image)
        output.extend(parse_pipeline_output(output_image, 'image'))
    except Exception as e:
        logger.exception("[ERROR] Failed in image pipeline: %s", e)

    return pd.DataFrame(output)


def load_input_data(scope: str, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads input CSV data for both text and image entries.

    Args:
        scope: Project scope (e.g., brand or report scope).
        source: Specific source identifier.

    Returns:
        Tuple of (df_text, df_image) DataFrames.
    """

    csv_filename = os.path.join(get_datafetch(), scope, source, STEP_2_TEXT_FILENAME)
    df_text = safe_read_csv(csv_filename)
    for field in REQUIRED_TEXT_COLUMNS:
        if field not in df_text.columns:
            raise ValueError(f"Missing expected column: {field}")

    csv_filename = os.path.join(get_datafetch(), scope, source, STEP_2_IMAGE_FILENAME)
    df_image = safe_read_csv(csv_filename)
    df_image.drop(labels=['websearch_text'], axis=1, inplace=True)

    return df_text, df_image


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', type=str, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--search_context_size', type=str, required=True)
    parser.add_argument('--websearch_model', type=str, required=True)

    return parser.parse_args()


def init_websearch_services(websearch_model: str, search_context_size: str) -> WebSearchService:
    """Initializes the web search service.

    Args:
        websearch_model: The OpenAI model name to use.
        search_context_size: Context size for the web search.

    Returns:
        An initialized instance of WebSearchService.

    Raises:
        EnvironmentError: If the OPENAI_API_KEY is missing.
    """

    credential_init()
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")
    client = OpenAI(api_key=openai_key)

    websearch_service = WebSearchService(client=client,
                                         search_context_size=search_context_size,
                                         model=websearch_model)

    return websearch_service


def build_extraction_pipeline(model_name: str) -> Runnable:
    """Constructs the extraction pipeline from model and prompt.

    Args:
        model_name: Name of the model to use.

    Returns:
        A LangChain Runnable pipeline.
    """

    model = model_activation(model_name=model_name)
    extract_prompt = build_extraction_prompt()
    extract_pipeline = extract_prompt | model | output_parser

    return extract_pipeline


def save_output_df(output_df: pd.DataFrame, scope: str, source: str):
    """Saves the processed output DataFrame to a CSV file.

    Args:
        output_df: Final DataFrame containing parsed data.
        scope: The scope used for data directory structure.
        source: The source used for data directory structure.
    """

    output_df.drop(labels=['if_strategy', 'text', 'signature'], axis=1, inplace=True)

    output_df.drop_duplicates(subset=['url'], inplace=True)
    output_df.reset_index(drop=True, inplace=True)

    filename = os.path.join(get_datafetch(), scope, source, STEP_3_FILENAME)
    output_df.to_csv(filename, index=False)


def main():
    """Main function to orchestrate the ETL pipeline."""
    args = parse_args()

    df_text, df_image = load_input_data(scope=args.scope, source=args.source)

    websearch_service = init_websearch_services(args.websearch_model, args.search_context_size)

    websearch_text_runnable = RunnableLambda(partial(websearch_text, websearch_engine=websearch_service))
    websearch_image_runnable = RunnableLambda(partial(websearch_image, websearch_engine=websearch_service))

    extract_pipeline = build_extraction_pipeline(model_name=args.model_name)

    websearch_text_pipeline_ = RunnablePassthrough.assign(text=websearch_text_runnable) | RunnablePassthrough.assign(
        results=extract_pipeline)

    websearch_image_pipeline_ = RunnablePassthrough.assign(text=websearch_image_runnable) | RunnablePassthrough.assign(
        results=extract_pipeline)

    output_df = run_pipeline(df_text=df_text, df_image=df_image, text_pipeline=websearch_text_pipeline_,
                             image_pipeline=websearch_image_pipeline_)

    save_output_df(output_df=output_df, scope=args.scope, source=args.source)


if __name__ == "__main__":

    main()

