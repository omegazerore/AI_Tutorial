from textwrap import dedent

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.logic import build_standard_chat_prompt_template


system_template = dedent("""
            You are a helpful AI assistant as a personal assistant of a trend manager.
            You always do the best you can and pay attention to details.
            """)

human_template = dedent("""
            Given a list of subtrends shown in {note} following the structure:
        
            **<SUBTREND>:**
        
            - **Definition:** <A clear and descriptive definition (2-3 sentences).>
            - **Examples:** <Relevant examples illustrating the subtrend.>
        
            Please help me identify if any of the subtrends matches:
        
            {subtrend}
        
            Output format instruction: {format_instructions}
            """)


class MatchResult(BaseModel):
    """Pydantic model for subtrend matching results."""
    result: str = Field(description="if there is a match, either `YES` or `NO`")
    reason: str = Field(description='Why you think they match or do not match')


output_parser = PydanticOutputParser(pydantic_object=MatchResult)
format_instructions = output_parser.get_format_instructions()


def build_prompt_template():
    """Builds a chat prompt template for trend analysis using LangChain.

    Returns:
        A LangChain chat prompt template ready for execution.
    """

    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                        "input_variables": ['note', 'subtrend'],
                        "partial_variables": {"format_instructions": format_instructions}}
              }

    return build_standard_chat_prompt_template(input_)


