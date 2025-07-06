"""Trend Week Report Prompt Builder

This module defines structured output models and builds a prompt template
for summarizing insights and generating recommended actions from content.
It uses LangChain's PydanticOutputParser and follows Google's Python style
guide and PEP 8 standards.
"""
from textwrap import dedent
from typing import List

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.logic.trend_week_report import build_standard_chat_prompt_template


class Action(BaseModel):
    """Represents a single recommended action derived from insights."""

    name: str = Field(description="recommended action")


class Finding(BaseModel):
    """Represents a key insight or observation, ideally supported by data or examples."""

    name: str = Field(description="A key insight or observation, ideally supported by example statistics or "
                                  "data points. This provides essential context to help others understand "
                                  "the basis for the recommended actions and the insights that led to them.")


class Result(BaseModel):
    """Structured output model containing insights and actions."""

    result: List[Action] = Field(
        description="A list of recommended actions associated with the insights"
    )
    key_findings: List[Finding] = Field(
        description="A list of key insights"
    )
    insight: str = Field(
        description="a short summary of insight provided by the report"
    )


# Parser to enforce output structure using the Result model
output_parser = PydanticOutputParser(pydantic_object=Result)
format_instructions = output_parser.get_format_instructions()


# System message template describing the AI assistant's role
system_template = dedent("""
    You are a highly capable AI assistant acting as the personal assistant to a trend manager.
    Your role is to provide detailed, accurate, and insightful support, always paying close attention to nuances and emerging developments.
    """)

# Human message template with formatting instructions
human_template = dedent("""
    Given the following content, please generate:
    1. A summarized version by clustering conceptually similar insights together.
    2. Provide a list of actions based on the generated insights.
    3. A short summary of the insights.

    **content:**

    {text}

    Each action follows this structure:
    ### Instructions:
    - Group actions that are conceptually similar based on their underlying insights.
    - Create a refined, summarized action for each group, ensuring clarity and coherence.

    **Output format:** {format_instructions}
    """)


def build_prompt_template():
    """Builds a standardized prompt template for summarizing insights and recommending actions.

    Returns:
        dict: A dictionary containing system and human templates with variables and formatting.
    """
    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                        "input_variables": ['text'],
                        "partial_variables": {"format_instructions": format_instructions}}
              }

    return build_standard_chat_prompt_template(input_)