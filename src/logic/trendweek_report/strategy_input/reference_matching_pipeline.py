"""Prompt builder for subtrend-insight matching in trend reports.

This module defines the prompt structure and output schema used to assess
whether a given note (insight) matches a specified subtrend, using LangChain's
chat prompt templating and Pydantic for output parsing.

The assistant role is positioned as a personal assistant to a trend manager.
"""
import json
import re
from textwrap import dedent
from typing import Literal

from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, chain
from pydantic import BaseModel, Field

from src.logic.trend_week_report import build_standard_chat_prompt_template

# System message template defining assistant persona
system_template = dedent("""
    You are a helpful AI assistant as a personal assistant of a trend manager.
    You always do the best you can and pay attention to details.""")

# Human message template that includes variables for the note and subtrend
human_template = dedent("""
    Given an insight show in:

    {note}

    Please help me identify if this content matches:

    {subtrend}

    output format instruction: {format_instructions}""")


class MatchResult(BaseModel):
    """Pydantic model defining the output structure for trend matching result.

    Attributes:
        result: A string indicating if the note matches the subtrend. Either 'YES' or 'NO'.
        reason: A textual explanation justifying the match or mismatch.
    """
    result: Literal["YES", "NO"] = Field(description="if there is a match, either `YES` or `NO`")
    reason: str = Field(description='Why you think they match or do not match')

output_parser_ = PydanticOutputParser(pydantic_object=MatchResult)
format_instructions = output_parser_.get_format_instructions()

@chain
def output_parser(kwargs):

    try:
        output_result = output_parser_.parse(kwargs.content)
        return output_result.model_dump()
    except Exception as e:
        match = re.search(r"\{.*\}", kwargs.content, re.DOTALL)
        group = match.group(0)
        return json.loads(group)

def build_prompt_template() -> Runnable:
    """Constructs a LangChain chat prompt template for subtrend-insight relevance assessment.

    Returns:
        A Runnable chat prompt that incorporates assistant persona and dynamic content
        for 'note' and 'subtrend', returning structured results using a Pydantic parser.
    """
    prompt_input = {"system": {"template": system_template},
                    "human": {"template": human_template,
                                "input_variables": ['note', 'subtrend'],
                                "partial_variables": {"format_instructions": format_instructions}}
                    }

    return build_standard_chat_prompt_template(prompt_input)