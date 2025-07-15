"""Main pipeline for verifying brand names using LangChain and websearch."""
import argparse
import logging
import os
import sys
from functools import partial
from typing import List, Dict, Tuple

import pandas as pd
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from textwrap import dedent

from src.initialization import model_activation
from src.io.path_definition import get_datafetch
from src.logic import build_standard_chat_prompt_template
from src.logic.product_news import STEP_3_FILENAME
from src.logic.product_news.websearch_service import activate_websearch_service, WebSearchService




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Output(BaseModel):
    """Pydantic output schema for brand comparison results."""
    name: bool = Field(description="True if the brands are the same, False otherwise")

# LangChain output parser
output_parser = PydanticOutputParser(pydantic_object=Output)
format_instructions = output_parser.get_format_instructions()


def build_extraction_prompt():
    """
    Builds a LangChain-compatible prompt for extracting boolean comparison.

    Returns:
        A LangChain prompt template.
    """
    system_template = (
        "You are a helpful assistant. Your task is to identify if the answer is true or false "
        "from a given input text"
    )

    human_template = "text: {text}\noutput format instruction: {format_instructions}"

    input_ = {"system": {"template": system_template,},
              "human": {"template": human_template,
                        "input_variables": ['text'],
                        "partial_variables": {"format_instructions": format_instructions}}}

    prompt_template = build_standard_chat_prompt_template(input_)

    return prompt_template


def run_websearch_query(kwargs, websearch_service):
    """
    Executes a brand comparison query via web search.

    Args:
        kwargs: Dictionary containing 'brand_text', 'brand_image', and 'country_code'.
        websearch_service: Initialized websearch service.

    Returns:
        A string containing the websearch result.
    """
    brand_text = kwargs['brand_text']
    brand_image = kwargs['brand_image']
    country_code = kwargs['country_code']

    if country_code == 'UK':
        country_code = 'GB'

    messages = [{"role": "user",
                 "content": dedent(f"""
                 Are brand {brand_text} and brand {brand_image} the same?
                 """)}]

    result = websearch_service.search(messages, country_code=country_code)

    return result


def image_row_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters image rows whose brands already exist in text rows.

    Args:
        df: Input DataFrame containing brand data.

    Returns:
        Filtered DataFrame containing image-based brand data only.
    """
    # if brands already exist in source_data_type text, ignore those in the source_data_type image
    brand_text = set(df[df['source_data_type'] == 'text']['brand'].str.lower())

    df_image_2_text = df[(df['source_data_type'] == 'image') & (df['brand'].str.lower().isin(brand_text))]
    # Remove all the rows whose brands extracted from images overlap brands extracted from text
    image_files = df_image_2_text['image_file_name'].unique()

    df_image = df[(~df['image_file_name'].isin(image_files)) & (df['source_data_type'] == 'image')]

    logger.info(f"Brands in text {brand_text}")
    logger.info(f"Brands in images {df_image['brand'].unique()}")

    return df_image


def create_batch(df_image: pd.DataFrame, df_text: pd.DataFrame) -> List[Dict]:
    """
    Creates a batch of image-text brand pairs for comparison.

    Args:
        df_image: DataFrame of image-based brand entries.
        df_text: DataFrame of text-based brand entries.

    Returns:
        List of dictionaries for pipeline input.
    """
    df_combined = pd.merge(df_image[['brand', 'country_code', 'strategy']],
                           df_text[['brand', 'country_code', 'strategy']],
                           on=['country_code', 'strategy'],
                           suffixes=('_image', '_text'),
                           how='left')

    df_combined.dropna(subset=['brand_text'], inplace=True)

    batch = df_combined.to_dict(orient='records')

    return batch


def prepare_data(scope: str, source: str) -> pd.DataFrame:
    """
    Loads input CSV file for the given scope and source.

    Args:
        scope: Data folder scope (e.g. 'cosmetics').
        source: Source name for input (e.g. 'trend').

    Returns:
        DataFrame with loaded data.
    """
    dir_ = os.path.join(get_datafetch(), scope, source)
    filename = os.path.join(dir_, STEP_3_FILENAME)
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found at {filename}")
        sys.exit("")


def initialize_services(model_name: str, websearch_model: str, search_context_size: str) -> Tuple[ChatOpenAI, WebSearchService]:
    """
    Initializes the language model and websearch service.

    Args:
        model_name: Name of the chat model.
        websearch_model: Name of the websearch model.
        search_context_size: Context size for websearch.

    Returns:
        Tuple of (ChatOpenAI model, WebSearchService).
    """
    model = model_activation(model_name=model_name)
    websearch_service = activate_websearch_service(
        model_name=websearch_model,
        search_context_size=search_context_size
    )
    return model, websearch_service


def build_pipeline(model, websearch_service) -> Runnable:
    """
    Constructs the LangChain processing pipeline.

    Args:
        model: Chat model to use.
        websearch_service: Initialized websearch service.

    Returns:
        A Runnable LangChain pipeline.
    """
    prompt_template = build_extraction_prompt()
    websearch_ = partial(run_websearch_query, websearch_service=websearch_service)

    step_1 = RunnablePassthrough.assign(text=websearch_)
    step_2 = RunnablePassthrough.assign(output=prompt_template | model | output_parser)

    return step_1 | step_2


def generate_batch(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Prepares input batches and filtered image/text data.

    Args:
        df: Original input DataFrame.

    Returns:
        Tuple of (text_df, image_df_filtered, batch list).
    """
    df_text = df[df['source_data_type'] == 'text'].drop_duplicates(subset=['brand'])
    df_image_filtered = image_row_selection(df).drop_duplicates(subset=['brand'])
    batch = create_batch(df_image=df_image_filtered, df_text=df_text)
    return df_text, df_image_filtered, batch


def run_pipeline(pipeline, batch: List[Dict]) -> pd.DataFrame:
    """
    Executes the processing pipeline on a batch.

    Args:
        pipeline: LangChain pipeline to use.
        batch: List of input dictionaries.

    Returns:
        DataFrame of parsed results.
    """
    results = pipeline.batch(batch)
    final_results = []
    for result in results:
        r = result.copy()
        output = r['output'].model_dump()
        del r['output']
        r.update(output)
        final_results.append(r)
    return pd.DataFrame(data=final_results)


def save_output(df_text: pd.DataFrame, df_image: pd.DataFrame, results_df: pd.DataFrame, scope: str, source: str):
    """
    Saves the final processed output as an Excel file.

    Args:
        df_text: DataFrame of text-based entries.
        df_image: DataFrame of image-based entries.
        results_df: Result DataFrame after pipeline run.
        scope: Scope folder for output.
        source: Source sheet name.
    """
    to_remove = results_df[results_df['name'] == False]['brand_image'].tolist()
    df_image_filtered = df_image[~df_image['brand'].isin(to_remove)]

    final_df = pd.concat([df_text, df_image_filtered], ignore_index=True)

    output_path = os.path.join(get_datafetch(), scope, "product_news.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name=source, index=False)


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


def main():

    args = parse_args()

    df = prepare_data(scope=args.scope, source=args.source)

    df_image = image_row_selection(df)

    model, websearch_model = initialize_services(model_name=args.model_name,
                                                 websearch_model=args.websearch_model,
                                                 search_context_size=args.search_context_size)

    pipeline = build_pipeline(model=model, websearch_service=websearch_model )

    df_text, df_image_filtered, batch = generate_batch(df)

    results_df = run_pipeline(pipeline=pipeline, batch=batch)

    save_output(df_text, df_image, results_df, args.scope, args.source)


if __name__ == "__main__":

    main()