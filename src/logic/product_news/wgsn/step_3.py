"""
EVY Technology Daily Defence Face Mousse Sonnencreme online kaufen
EVY TECHNOLOGY UV / Heat Hair Mousse Sonnenspray ✔️ online kaufen | DOUGLAS
[TIRTIR] Off The Sun Air Mousse SPF 50+ PA++++ 100ml - COCOMO
BREAD BEAUTY SUPPLY - Hair-Foam: Curling Mousse | Ulta Beauty
NUSE Mousse Liptual – Ma Petite Coree
The Dream Collection Whipped Shower Foam - Reichhaltiger Duschschaum | RITUALS
Mousse de Banho Baunilha | Espuma hidratante para um banho perfumado
"""
import ast
import argparse
import os
from typing import List

import pandas as pd
from textwrap import dedent
from openai import OpenAI
from langchain_core.runnables import chain, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.initialization import model_activation, credential_init
from src.io.path_definition import get_datafetch
from src.logic.product_news.websearch_service import WebSearchService
from src.logic.trend_week_report import build_standard_chat_prompt_template
from src.logic.product_news import STEP_2_TEXT_FILENAME, STEP_2_IMAGE_FILENAME, STEP_3_FILENAME

websearch_service_text: WebSearchService = None
websearch_service_image: WebSearchService = None


class Source(BaseModel):
    content: str = Field(description="description of the product.")
    url: str = Field(description="url of the content source.")
    name: str = Field(description="The product, including the brand name.")
    brand: str = Field(description="the brand name")


class Output(BaseModel):
    name: List[Source] = Field(description="A list of content and url")


output_parser = PydanticOutputParser(pydantic_object=Output)
format_instructions = output_parser.get_format_instructions()


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
    except pd.errors.EmptyDataError:
        print(f"[ERROR] File is empty: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
    return pd.DataFrame()


@chain
def websearch_text(kwargs):

    global websearch_service_text

    strategy = kwargs['strategy']
    product = kwargs['name']
    brand = kwargs['brand']
    country_code = kwargs['country_code']
    if country_code == 'UK':
        country_code = "GB"

    messages = [{"role": "user",
                 "content": dedent(f"""
                 What is the product {brand} {product} under the concept {strategy}?
                 Please find the best match and the corresponding product page, in which one can order the product.
                 Return only one product page.
                 """)}]

    result = websearch_service_text.search(messages, country_code=country_code)

    return result


@chain
def websearch_image(kwargs):

    global websearch_service_image

    strategy = kwargs['strategy']
    context = kwargs['text']
    brand = kwargs['brand']
    country_code = kwargs['country_code']
    if country_code == 'UK':
        country_code = "GB"
    if pd.isnull(kwargs['country_code']):
        country_code = None

    messages = [{"role": "user",
                 "content": f"What is the cosmetic or skin care product under the concept {strategy} with brand {brand}?\n\n"
                            f"The concept has the following context: {context}\n\n"
                            f"Please provide me the page to the product of the brand.\n"
                            f"If the exact product cannot be found, please give me products with similar concept within {brand}.\n"
                            f"Ideally from the official website of the brand {brand}."
                 }]

    result = websearch_service_image.search(messages, country_code=country_code)

    return result


def build_extraction_prompt():

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


def main(scope: str, source: str, model_name: str, search_context_size: str,
         websearch_model: str):

    global websearch_service_text, websearch_service_image

    model = model_activation(model_name=model_name)

    credential_init()
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")
    client = OpenAI(api_key=openai_key)

    websearch_service_text = WebSearchService(client=client,
                                              search_context_size=search_context_size,
                                              model=websearch_model)
    websearch_service_image = WebSearchService(client=client,
                                               search_context_size=search_context_size,
                                               model=websearch_model)

    csv_filename = os.path.join(get_datafetch(), scope, source, STEP_2_TEXT_FILENAME)
    df_text = safe_read_csv(csv_filename)
    required_fields = ['name', 'if_strategy', 'page_number', 'file', 'brand', 'country_code']
    for field in required_fields:
        if field not in df_text.columns:
            raise ValueError(f"Missing expected column: {field}")

    csv_filename = os.path.join(get_datafetch(), scope, source, STEP_2_IMAGE_FILENAME)
    df_image = safe_read_csv(csv_filename)
    df_image.drop(labels=['websearch_text'], axis=1, inplace=True)

    extract_prompt = build_extraction_prompt()
    extract_pipeline = extract_prompt | model | output_parser

    websearch_text_pipeline_ = RunnablePassthrough.assign(text=websearch_text) | RunnablePassthrough.assign(
        results=extract_pipeline)

    websearch_image_pipeline_ = RunnablePassthrough.assign(text=websearch_image) | RunnablePassthrough.assign(
        results=extract_pipeline)

    output = []

    batch = df_text[df_text['if_strategy']].to_dict(orient="records")
    # for _, row in df_text.iterrows():
    #     result_as_dict = row.to_dict()
    #     if row['if_strategy']:
    #         try:
    #             products = ast.literal_eval(row['products'])
    #         except (ValueError, SyntaxError) as e:
    #             print(f"Error parsing products for row {row}: {e}")
    #             products = []
    #         for product in products:
    #             input_ = result_as_dict.copy()
    #             input_['product'] = product
    #             batch.append(input_)

    output_text = websearch_text_pipeline_.batch(batch)
    for c in output_text:
        results = c['results'].name
        del c['results']
        for result in results:
            c_copy = c.copy()
            c_copy.update(result.model_dump())
            c_copy["source_data_type"] = 'text'
            output.append(c_copy)

    df_metadata = df_text.drop_duplicates(subset=['page_number', 'file'])[['page_number', 'file', 'strategy', 'text']]

    df_image = df_image.merge(df_metadata, on=['page_number', 'file'])
    # df_image = df_image[df_image['image_file_name']=='figure-10-3.jpg']
    df_image.dropna(subset=['brand'], inplace=True)

    batch = df_image.to_dict(orient="records")
    output_image = websearch_image_pipeline_.batch(batch)

    for c in output_image:
        results = c['results'].name
        del c['results']
        for result in results:
            c_copy = c.copy()
            c_copy.update(result.model_dump())
            c_copy["source_data_type"] = 'image'
            output.append(c_copy)

    output_df = pd.DataFrame(output)

    output_df.drop(labels=['if_strategy', 'text', 'signature'], axis=1, inplace=True)

    output_df.drop_duplicates(subset=['url'], inplace=True)
    output_df.reset_index(drop=True, inplace=True)

    filename = os.path.join(get_datafetch(), scope, source, STEP_3_FILENAME)
    output_df.to_csv(filename, index=False)


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

