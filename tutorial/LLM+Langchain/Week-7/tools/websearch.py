from textwrap import dedent

from openai import OpenAI, AsyncOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.initialization import credential_init

credential_init()


client = OpenAI()
async_client = AsyncOpenAI()



class Inputs(BaseModel):
    query: str = Field(description="User query")
    country_code: str = Field(description="ISO 3166-1 alpha-2 suggested by the language of the user query")


class SearchTool(BaseTool):

    input_output_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Inputs)
    input_format_instructions: str = input_output_parser.get_format_instructions()
    
    name:str = "websearch tool"
    description_template:str = dedent("""
    Currently it is 2025.    
    Use this tool to collect information from the internet, when you are not sure you know the answer.
    The input contains the user's question `query` and the ISO 3166-1 alpha-2 `country_code` inferred from the user's language.
    input format instructions: {input_format_instructions}
    """)

    description: str = description_template.format(input_format_instructions=input_format_instructions)

    def _build_messages_and_opts(self, query: str):
        """Shared logic for sync + async"""
        
        input_ = self.input_output_parser.parse(query)
        query = input_.query
        country_code = input_.country_code

        tool = {"type": "web_search",
                         "user_location":{
                             "type": "approximate",
                             "country": country_code,
                         },
                        "search_context_size": "medium"
                        }
        return query, tool
        
    def _run(self, query):
        
        query, tool = self._build_messages_and_opts(query)

        response = client.responses.create(
                    model="gpt-4o-mini",
                    tools=[tool],
                    tool_choice="auto",
                    input=query)

        
        return response.output_text
    
    async def _arun(self, query: str):
        
        query, tool = self._build_messages_and_opts(query)

        response = await async_client.responses.create(
            model="gpt-4o-mini",
            tools=[tool],
            tool_choice="auto",
            input=query
        )

        return response.output_text