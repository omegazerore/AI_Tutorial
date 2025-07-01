from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.logic import build_standard_chat_prompt_template


system_template = ("You are a helpful AI assistant as a personal assistant of a "
                   "trend manager.\n"
                   "You always do the best you can and pay attention to details.")

human_template = ("Please create a note based given the following context:\n"
                  "{text}.\n\n"
                  "The note should contain the following MEGATRENDS:\n\n"
                  "1. Society: Related to consumer behavior, populations, or generational "
                  "trends.\n"
                  "2. Technology: Includes AI, AR, beauty tech, and other emerging "
                  "technologies.\n"
                  "3. Environment: Focuses on sustainability, ecological impact, and "
                  "green innovation.\n"
                  "4. Industry: Covers movements in skincare, color cosmetics, personal "
                  "care, and beauty innovation.\n\n"
                  "and the associated SUBTRENDS.\n\n"
                  "For each SUBTREND, the trend manager needs:\n"
                  "• A short definition or description of the subtrend.\n"
                  "• Examples of products, brands, or innovations mentioned in the "
                  "reports.\n"
                  "Please cluster the identified subtrends under the appropriate megatrend "
                  "category, ensuring the analysis is concise and actionable. If specific "
                  "examples are unavailable, highlight the general direction of the trend "
                  "instead\n\n"
                  "output format instruction: {format_instructions}")


class SubTrend(BaseModel):

    name: str = Field(description=("The name of the subtrend"))
    definition: str = Field(description=("The definition of the subtrend"))
    examples: str = Field(description=("Examples of the subtrend"))


class MegaTrend(BaseModel):
    name: str = Field(description=("The name of the megatrends, must be one of:\n\n"
                     "- Society: Related to consumer behavior, populations, or generational trends.\n"
                     "- Technology: Includes AI, AR, beauty tech, and other emerging technologies.\n"
                     "- Environment: Focuses on sustainability, ecological impact, and green innovation.\n"
                     "- Industry: Covers movements in skincare, color cosmetics, personal care, and beauty innovation."))

    content: List[SubTrend] = Field(description="A list of Subtrend content")


class Trends(BaseModel):
    trends: List[MegaTrend] = Field(description="A list of MegaTrend content")


def build_prompt_template(kwargs: Optional=None):

    output_parser = PydanticOutputParser(pydantic_object=Trends)
    format_instructions = output_parser.get_format_instructions()

    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                  "input_variables": ['text'],
                  "partial_variables": {"format_instructions": format_instructions}}
            }

    return build_standard_chat_prompt_template(input_), output_parser

