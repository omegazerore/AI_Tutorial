import os
from typing import List, Dict
from datetime import datetime, timedelta
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain, RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from openai.types.responses.response import Response
from pydantic import BaseModel, Field

from src.initialization import credential_init
from src.logic.trend_week_report import build_standard_chat_prompt_template
from src.logic.trend_week_report.deep_search import OpenAIWebSearch


class URL(BaseModel):
    name: str = Field(description='source/url/website')


class Example(BaseModel):
    brand: str = Field(description="brand name")
    content: str = Field(description='content. Content under the same brand should be aggregated.')
    urls: List[URL] = Field(description='A list of source/url/website')


class Examples(BaseModel):
    result: List[Example] = Field(description="A list of examples")


class SearchQueryAssistant:

    credential_init()

    date_end = datetime.now()
    date_begin = date_end - timedelta(days=365)

    date_end = date_end.strftime("%Y-%m")
    date_begin = date_begin.strftime("%Y-%m")

    print(f"date_begin: {date_begin}\ndate_end: {date_end}")

    output_parser = PydanticOutputParser(pydantic_object=Examples)
    format_instructions = output_parser.get_format_instructions()

    def __init__(self, model: str):

        self._model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                                 model_name=model, temperature=0)

        self.prompt_pipeline = self._build_query_generation_pipeline()

        self.parsing_pipeline = self._build_search_result_parsing_pipeline()

    def _build_query_generation_pipeline(self):

        human_template = ("Identify the most relevant service, campaign, or product that aligns with the given description: {description}, "
                          "ensuring it is associated with the brands mentioned in the provided example: {example}.\n"
                          "Limit the results to the time frame between {date_begin} and {date_end}")

        input_ = {"human": {"template": human_template,
                            "input_variables": ['description', 'example'],
                            "partial_variables": {"date_begin": self.date_begin,
                                                  "date_end": self.date_end}}}

        chat_prompt_template = build_standard_chat_prompt_template(input_)

        return chat_prompt_template|self._ai_message_content

    @staticmethod
    @chain
    def _ai_message_content(chat_prompt):

        return chat_prompt.messages[0].content

    def _build_search_result_parsing_pipeline(self):

        human_template = (
            "Parse {content}\n\n**Output format:** {format_instructions}")

        input_ = {"human": {"template": human_template,
                            "input_variables": ['content'],
                            "partial_variables": {"format_instructions": self.format_instructions}}}

        chat_prompt_template = build_standard_chat_prompt_template(input_)

        return chat_prompt_template|self._model|self.output_parser

@chain
def response_to_output_text(response: Response) -> str:

    return response.choices[0].message.content


def service(inputs: List[Dict]):

    example_search = SearchQueryAssistant(model='gpt-4o-mini')
    openai_websearch = OpenAIWebSearch(model="o3-mini", temperature=0)

    parsing_step = RunnablePassthrough.assign(final_output=itemgetter("context") | response_to_output_text | example_search.parsing_pipeline)

    pipeline_ = RunnablePassthrough.assign(question=example_search.prompt_pipeline) | openai_websearch.pipeline | parsing_step

    return pipeline_.batch(inputs)


if __name__ == "__main__":

    example_search = SearchQueryAssistant(model='gpt-4o-mini')
    openai_websearch = OpenAIWebSearch(model="o3-mini", temperature=0)

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
