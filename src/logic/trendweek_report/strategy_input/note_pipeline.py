"""Module for building a structured chat prompt to generate trend insights using LangChain and Pydantic.

This module provides a chat template that guides an AI assistant to extract key megatrend insights
and recommended actions from provided context, specifically for a trend manager in the beauty industry.
"""

from textwrap import dedent
from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from src.logic.trend_week_report import build_standard_chat_prompt_template


# System-level instruction template for AI behavior
system_template = dedent("""
    You are a highly capable AI assistant acting as the personal assistant to a trend manager.
    Your role is to provide detailed, accurate, and insightful support, always paying close attention to nuances and emerging developments.

    The following technical terms may appear in context:
    1. SBTi (Science Based Targets initiative): A global effort that helps companies set climate goals aligned with scientific consensus to reduce carbon dioxide emissions.
    2. APAC (Asia-Pacific): A critical region—including China, Japan, and South Korea—known for driving global beauty and innovation trends.
""")

# Human prompt template guiding the AI to analyze megatrends
human_template = dedent("""
    Please create a note based on the following context:
    {text}.

    The note should contain the following MEGATRENDS:

    1. Beauty Categories & Consumer Trends: Growth or decline of specific categories, formats, 
       or routines, Shifts in consumer behaviour or beauty expectations.
    2. Sustainability: True consumer demand for sustainability – by region or demographic, Relevance of standards and frameworks.
       e.g.: Science Based Targets initiative – helps companies set climate goals aligned with science to cut carbon dioxide emissions.
    3. Societal Shifts: Consumer and employee response to societal uncertainty or instability, Role and 
       perception of DEI - including any backlash, Evolving expectations toward brands as trust in institutions 
       erodes, The future of work and how we will work.
    4. Market & Global Development: Regional differences in opportunity (Europe, North America, Latin America, APAC, 
       Middle East), Trends in distribution: e-commerce vs. offline, Where to invest or pull back.
    5. New Ways of Working: Strategic relevance of hybrid work, adaptability, continuous learning, Emerging leadership models.

    For each MEGATREND, please include:
    • **Key Insights**: Clear, concise observations that reflect meaningful shifts or patterns from the context.
      - Where possible, support insights with relevant statistics or data points to strengthen the analysis.
    • **Recommended Actions**: Strategic next steps or opportunities that are:
      - Forward-looking (relevant from 2025–2028)
      - Specific to the color cosmetics and skincare segments
      - Practical, clearly connected to the insights, and clustered under the corresponding megatrend.

    Ensure the analysis is well-structured, evidence-based, and concise.

    Output format instruction: {format_instructions}
""")


class Action(BaseModel):
    """Represents a recommended strategic action derived from trend analysis."""

    name: str = Field(description=("recommended action"))


class MegaTrend(BaseModel):
    """Represents a single megatrend with associated insights and strategic actions."""

    megatrend_instruction: str = dedent("""
        The name of the megatrends must be one of:

        - `Beauty Categories & Consumer Trends`
        - `Sustainability`
        - `Societal Shifts`
        - `Market & Global Development`
        - `New Ways of Working`
    """)

    name: Literal["Beauty Categories & Consumer Trends",
                  "Sustainability",
                  "Societal Shifts",
                  "Market & Global Development",
                  "New Ways of Working"] = Field(description=megatrend_instruction)

    actions: List[Action] = Field(description="A list of insight based recommended actions")
    insight: str = Field(description="Key insights")


class Trends(BaseModel):
    """Represents a structured list of megatrend analyses."""

    trends: List[MegaTrend] = Field(description="A list of MegaTrend content")


def build_prompt_template():
    """Builds the chat prompt template and output parser.

    Returns:
        tuple:
            A tuple containing:
            - prompt_template (ChatPromptTemplate): Compiled prompt template.
            - output_parser (PydanticOutputParser): Parser for validating responses.
    """

    output_parser = PydanticOutputParser(pydantic_object=Trends)
    format_instructions = output_parser.get_format_instructions()

    prompt_input = {"system": {"template": system_template},
                    "human": {"template": human_template,
                        "input_variables": ['text'],
                        "partial_variables": {"format_instructions": format_instructions}}
                }

    return build_standard_chat_prompt_template(prompt_input), output_parser

