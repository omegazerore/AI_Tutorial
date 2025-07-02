"""Search query assistant and web search parsing pipeline.

This module defines a `SearchQueryAssistant` class that builds query-generation
and parsing pipelines using LangChain and OpenAI models, and provides a
`service` function to batch-process input queries with web search results.

Classes:
    URL: A pydantic model representing a web source.
    Example: A pydantic model representing a brand example with content and URLs.
    Examples: A pydantic model wrapping a list of `Example` entries.
    SearchQueryAssistant: Encapsulates prompt and parsing pipelines for trend analysis.

Functions:
    response_to_output_text(response): Extracts the response text from an OpenAI response object.
    service(inputs, search_context_size, search_model): Executes the query and parsing pipeline.

"""
import os
from datetime import datetime, timedelta
from operator import itemgetter
from textwrap import dedent
from typing import Dict, List

from langchain_core.runnables import chain, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from openai.types.responses.response import Response
from pydantic import BaseModel, Field

from src.initialization import credential_init, model_activation
from src.logic import build_standard_chat_prompt_template
from src.logic.trendweek_report.future_vision.utils_websearch.deep_websearch_service import OpenAIWebSearch


class URL(BaseModel):
    """Model representing a single source URL."""

    name: str = Field(description='source/url/website')


class Example(BaseModel):
    """Model representing an example brand with associated content and sources."""


    brand: str = Field(description="brand name")
    content: str = Field(description='content. Content under the same brand should be aggregated.')
    urls: List[URL] = Field(description='A list of source/url/website')


class Examples(BaseModel):
    """Model representing a list of examples."""

    result: List[Example] = Field(description="A list of examples")


class SearchQueryAssistant:
    """Assists in building and executing search query pipelines using OpenAI and LangChain.

    Attributes:
        date_begin: Beginning of the time range for search queries (format YYYY-MM).
        date_end: End of the time range for search queries (format YYYY-MM).
        _credentials_initialized: Flag indicating whether credentials are initialized.
    """

    date_begin: str = None
    date_end: str = None
    _credentials_initialized: bool = False

    @classmethod
    def initialize(cls):
        """Initializes credentials and date range once for all instances."""
        if not cls._credentials_initialized:
            credential_init()
            cls._credentials_initialized = True

        now = datetime.now()
        cls.date_end = now.strftime("%Y-%m")
        cls.date_begin = (now - timedelta(days=365)).strftime("%Y-%m")

    def __init__(self, model: str):
        """Initializes the search assistant with the specified model.

        Args:
            model: Name of the OpenAI model to use.
        """
        self.initialize()  # Ensures class-level setup happens once

        self.output_parser = PydanticOutputParser(pydantic_object=Examples)
        self.format_instructions = self.output_parser.get_format_instructions()

        self._model = model_activation(model_name=model)

        self.prompt_pipeline = self._build_query_generation_pipeline()

        self.parsing_pipeline = self._build_search_result_parsing_pipeline()

    def _build_query_generation_pipeline(self):
        """Constructs the prompt pipeline for generating queries from descriptions.

        Returns:
            A runnable prompt pipeline.
        """
        human_template = dedent("""
            Identify the most relevant service, campaign, or product that aligns with the given description: {description},
            ensuring it is associated with the brands mentioned in the provided example: {example}.
            Limit the results to the time frame between {date_begin} and {date_end}.
        """)

        input_ = {"human": {"template": human_template,
                            "input_variables": ['description', 'example'],
                            "partial_variables": {
                                "date_begin": self.date_begin,
                                "date_end": self.date_end
                            }
                        }
                  }

        chat_prompt_template = build_standard_chat_prompt_template(input_)
        return chat_prompt_template|self._ai_message_content

    @staticmethod
    @chain
    def _ai_message_content(chat_prompt):
        """Extracts message content from AI response.

        Args:
            chat_prompt: The chat prompt template.

        Returns:
            Content string from the first AI message.
        """
        return chat_prompt.messages[0].content

    def _build_search_result_parsing_pipeline(self):
        """Constructs the pipeline for parsing search results using structured format.

        Returns:
            A runnable parsing pipeline.
        """
        human_template = (
            "Parse {content}\n\n**Output format:** {format_instructions}")

        input_ = {
            "human": {
                "template": human_template,
                "input_variables": ['content'],
                "partial_variables": {"format_instructions": self.format_instructions}
            }
        }

        chat_prompt_template = build_standard_chat_prompt_template(input_)

        return chat_prompt_template|self._model|self.output_parser


@chain
def response_to_output_text(response: Response) -> str:
    """Extracts content text from the OpenAI response object.

    Args:
        response: OpenAI response object.

    Returns:
        The content string of the first message.
    """
    return response.choices[0].message.content


def service(inputs: List[Dict], search_context_size, search_model):
    """Executes the end-to-end query and parsing pipeline.

    Args:
        inputs: List of dictionaries with 'description' and 'example'.
        search_context_size: Integer specifying the number of tokens or content size for web search.
        search_model: The search model to use for retrieving results.

    Returns:
        A list of parsed outputs for each input.
    """
    example_search = SearchQueryAssistant(model='gpt-4o-mini')
    openai_websearch = OpenAIWebSearch(model="o3-mini",
                                       search_context_size=search_context_size,
                                       search_model=search_model)

    parsing_step = RunnablePassthrough.assign(final_output=itemgetter("context") | response_to_output_text | example_search.parsing_pipeline)

    pipeline_ = RunnablePassthrough.assign(question=example_search.prompt_pipeline) | openai_websearch.pipeline | parsing_step

    return pipeline_.batch(inputs)


if __name__ == "__main__":

    example_search = SearchQueryAssistant(model='gpt-4o-mini')
    openai_websearch = OpenAIWebSearch(model="o3-mini")

    pipeline_ = RunnablePassthrough.assign(question=example_search.prompt_pipeline)|openai_websearch.pipeline|RunnablePassthrough.assign(final_output=itemgetter("context")|response_to_output_text|example_search.parsing_pipeline) #example_search.pipeline #

    description=("A growing awareness among consumers regarding the ethical implications of their purchases, leading to "
                 "a preference for brands that align with their values, including sustainability and inclusivity.")
    example=("Brands like Lush, The Body Shop, Fenty Beauty, and Glossier are known for their commitment to ethical "
             "sourcing, transparency, and diverse representation.")

    output = pipeline_.invoke({"description": description,
                               "example": example})

    print("\n********************************\n")

    for k in output['final_output'].result:
        print(k, end="\n")
