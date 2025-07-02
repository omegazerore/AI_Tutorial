"""
The API runs as expected.
"""
from textwrap import dedent
from functools import partial

from langchain_core.runnables import chain, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.initialization import model_activation
from src.logic import build_standard_chat_prompt_template
from src.logic.product_news.websearch_service import activate_websearch_service


class WebSearchResult(BaseModel):

    name: str = Field(description=("optimized search query"))


@chain
def query_to_message(kwargs):

    system_prompt = dedent("""
        You are an AI assistant that uses a web search tool when up-to-date, local, or specific information is required. 
        Before using the web, rely on your internal knowledge to respond to general questions. Use the web only when:

        - The user asks for recent or breaking news, current events, prices, schedules, or availability.
        - The question requires local details such as nearby businesses, services, or events.
        - The topic is niche, obscure, or likely to have changed since 2023.

        When using web search:
        - Perform a search and extract accurate, concise, and relevant information from trusted sources.
        - Summarize or synthesize rather than copy-paste.
        - Always explain that the information is from a web search if it's relevant to user trust or context.
        - If a search fails to return useful results, tell the user clearly.

        Do not hallucinate or guess when fresh information is required. Stay grounded and neutral. Be efficient, helpful, and honest.
        """)

    messages = [{"role": "system",
                 "content": system_prompt},
                {"role": "user",
                 "content": kwargs.name}]

    return messages


class OpenAIWebSearch:

    def __init__(self, model: str):

        """
        output: response.output_text
        citations: response.output[1].content[0].annotations
        """

        self._model = model_activation(model_name=model)

        websearch_service = activate_websearch_service(model_name="gpt-4o-search-preview",
                                                       search_context_size="high")

        output_parser, format_instructions = self._build_parser()

        search_partial = partial(websearch_service.search_with_annotation, country_code=None)

        self.web_search_pipeline = RunnableLambda(search_partial)#openai_web_search_fn
        self.query_generation_pipeline = self._build_query_generation_pipeline(format_instructions=format_instructions)

        self.pipeline = RunnablePassthrough.assign(context=self.query_generation_pipeline|
                                                           output_parser|
                                                           query_to_message|
                                                           self.web_search_pipeline
                                                   )

    def _build_query_generation_pipeline(self, format_instructions: str):

        input_ = {"system": {"template": ("You are a highly intelligent AI assistant specialized in refining and optimizing search queries.\n"
                                          "Your goal is to generate precise, well-structured, and effective web search queries that maximize "
                                          "relevant and accurate results.")},
                  "human": {"template": ("Help me generate an optimized web search query based on the following question: {question}.\n"
                                         "Ensure the query is clear, specific, and structured to retrieve the most relevant information efficiently.\n"
                                         "output format instructions: {format_instructions}"),
                            "input_variables": ['question'],
                            "partial_variables": {"format_instructions": format_instructions}}
                  }

        chat_prompt_template = build_standard_chat_prompt_template(input_)

        return chat_prompt_template|self._model

    @staticmethod
    def _build_parser():

        output_parser = PydanticOutputParser(pydantic_object=WebSearchResult)
        format_instructions = output_parser.get_format_instructions()

        return output_parser, format_instructions


if __name__ == "__main__":

    # credential_init()
    #
    # client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    #
    # response =  client.responses.create(
    #     model="gpt-4o",
    #     tools=[{"type": "web_search_preview"}],
    #     input="Give me a market wrap on the Hong Kong market for Feb 2025."
    # )
    #
    # print(response.output_text)

    # response.output[1].content[0].annotations

    websearch = OpenAIWebSearch(model='gpt-4o-mini')

    # output = websearch.pipeline.invoke({"question": "Give me a market wrap on the Hong Kong market for Feb 2025."})

    output = websearch.pipeline.invoke({"question": "What is the product with GTIN/Barcode = 4049639444607?"})

    print(output['context'])

    # print(output['context'].choices[0].message.content)
    #
    # print(f"{output['context'].choices[0].message.annotations}")