"""Trend summarization and recommendation prompt builder.

This module defines data models for trend analysis and constructs a chat prompt
template used to generate summaries with strategic recommendations based on
provided context. It integrates Pydantic models for structured output and uses
LangChain's output parser and template builder.

Classes:
    Recommendation: Represents strategic recommendations for a trend.
    Trend: Represents a trend with explanation and related recommendations.
    Trends: A wrapper for a list of Trend objects.

Functions:
    build_prompt_template: Constructs and returns a chat prompt template.

"""

from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from textwrap import dedent

from src.logic.trend_week_report import build_standard_chat_prompt_template


class Recommendation(BaseModel):
    """Represents a strategic recommendation related to a trend.

    Attributes:
        focus_area: Key area of strategic focus.
        portfolio_action: Recommended action related to product portfolio.
        regional_opportunity: Suggested regional market opportunity.
    """
    focus_area: str = Field(description="focus area")
    portfolio_action: str = Field(description="portfolio action")
    regional_opportunity: str = Field(description="regional opportunity")


class Trend(BaseModel):
    """Represents a single trend and its associated recommendations.

    Attributes:
        name: Name of the trend.
        explanation: Brief explanation of why this trend is important.
        recommendations: List of related strategic recommendations.
    """
    name: str = Field(description="The most relevant trend")
    explanation: str = Field(description="A brief 'Why this matters' explanation")
    recommendations: List[Recommendation] = Field(description="A list of recommendations of clear strategic "
                                                              "recommendation (e.g., focus area, portfolio action, "
                                                              "regional opportunity")


class Trends(BaseModel):
    """Container for a list of trends."""
    trends: List[Trend] = Field(description="A list of trend")


# Output parser to enforce structured output format using Pydantic
output_parser = PydanticOutputParser(pydantic_object=Trends)
format_instructions = output_parser.get_format_instructions()

# System-level instructions for the AI assistant
system_template = dedent("""
    You are a highly capable AI assistant acting as the personal assistant to a trend manager.
    Your role is to provide detailed, accurate, and insightful support, always paying close attention 
    to nuances and emerging developments.
""")

# Human-readable template for input context and summarization instructions
human_template = dedent("""
    Please generate a summary based on the context provided below:
    {text}

    The summary should highlight the top 2–3 trends that have the most significant business impact 
    within the skincare and color cosmetics categories.

    For each identified trend, include the following:
    1. A concise explanation of *why this trend matters*.
    2. A clear strategic recommendation, such as a key focus area, portfolio action, or regional opportunity.

    Please follow this output format: {format_instructions}
""")


def build_prompt_template():
    """Builds a chat prompt template for trend summarization.

    Returns:
        PromptTemplate: A LangChain chat prompt template instance with system and human messages.
    """
    prompt_input = {"system": {"template": system_template},
                    "human": {"template": human_template,
                        "input_variables": ['text'],
                        "partial_variables": {"format_instructions": format_instructions}}
                    }

    return build_standard_chat_prompt_template(prompt_input)