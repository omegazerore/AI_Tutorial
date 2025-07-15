"""Extracts text signatures from images and maps them to brand identities.

This module provides:

- `SignatureExtraction`: Extracts text signatures from images using a vision-language model.
- `Signature2Brand`: Maps extracted signatures to cosmetic/skin-care brand names and country codes
  using web search and a language model.

Dependencies:
    - LangChain
    - OpenAI
    - `src.logic` package (internal logic components)
"""
import logging
import time
from typing import List, Dict, Tuple, Optional
from operator import itemgetter

from openai import OpenAIError
from textwrap import dedent
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, chain, Runnable
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.image import ImagePromptTemplate
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field

from src.initialization import model_activation
from src.logic import build_standard_chat_prompt_template
from src.logic.product_news import image_to_base64, build_pipeline
from src.logic.product_news.websearch_service import WebSearchService
from src.logic.product_news import MAX_CONCURRENCY


IMAGE_PATH_KEY = "image_path"
SIGNATURE_KEY = "signature"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignatureOutput(BaseModel):
    """Pydantic model representing the signature extraction result."""
    name: str = Field(description="The signature on the image")


class BrandOutput(BaseModel):
    """Pydantic model representing the brand name derived from signature."""
    brand: str = Field(description="The brand")
    country_code: str = Field(description="ISO 3166-1 alpha-2 of the country of the brand")


class SignatureExtraction:
    """Extracts handwritten or visual signatures from image files.

    This class initializes a LangChain-compatible image-to-text pipeline that uses a
    vision-language model to extract signature text from provided image paths.
    """

    def __init__(self, model_name: str):
        """Initializes the signature extraction pipeline with a given model.

        Args:
            model_name (str): Name of the vision-language model to activate.
        """
        logger.info("Initializing SignatureExtraction")

        # Error handling is in the function model_activation
        model = model_activation(model_name)

        image_to_base64_pipeline = RunnableLambda(image_to_base64)

        output_parser, self.format_instructions = self._build_signature_parser()
        prompt_template = self.build_image_caption_prompt_template()

        step_1 = RunnablePassthrough.assign(image_str=itemgetter("image_path") | image_to_base64_pipeline)
        step_2 = RunnablePassthrough.assign(signature=prompt_template|model|output_parser|self.extract_name_field)

        self.pipeline = build_pipeline([step_1, step_2])

    def build_image_caption_prompt_template(self) -> ChatPromptTemplate:
        """Constructs a LangChain chat prompt for image signature captioning.

        Returns:
            ChatPromptTemplate: A chat prompt with both text and image components.
        """
        text_prompt_template = PromptTemplate(template="Please extract the signature on the image.\n"
                                                       'Output format instruction: {format_instructions}',
                                              partial_variables={"format_instructions": self.format_instructions})
        image_prompt_template = ImagePromptTemplate(template={"url": 'data:image/jpeg;base64,{image_str}'},
                                                    input_variables=['image_str'])

        human_message_template = HumanMessagePromptTemplate(
            prompt=[text_prompt_template,
                    image_prompt_template],
        )

        prompt_template = ChatPromptTemplate.from_messages([human_message_template])

        return prompt_template

    @staticmethod
    def _build_signature_parser() -> Tuple[PydanticOutputParser, str]:
        """Builds a parser for structured signature output.

        Returns:
            Tuple[PydanticOutputParser, str]: Output parser and format instructions.
        """
        output_parser = PydanticOutputParser(pydantic_object=SignatureOutput)
        format_instructions = output_parser.get_format_instructions()

        return output_parser, format_instructions

    @chain
    @staticmethod
    def extract_name_field(pydantic_object) -> str:
        """Extracts the 'name' field from a parsed object.

        Args:
            pydantic_object (BaseModel): A Pydantic object with a `name` field.

        Returns:
            str: Extracted name.
        """
        return pydantic_object.name

    def batch(self, batch: List[Dict[str, str]], max_concurrency: Optional[int]=None) -> List[Dict[str, str]]:
        """Runs the signature extraction pipeline on a batch of images.

        Args:
            batch (List[Dict[str, str]]): List of items containing 'image_path'.
            max_concurrency (Optional[int]): Maximum number of concurrent model executions.

        Returns:
            List[Dict[str, str]]: List of results with extracted signature names.

        Raises:
            ValueError: If 'image_path' is missing in any batch item.
            OpenAIError: If model inference fails.
        """
        logger.info(f"Processing batch with {len(batch)} items")
        start_time = time.time()

        for item in batch:
            if IMAGE_PATH_KEY not in item:
                logger.warning("Missing required key: %s", IMAGE_PATH_KEY)
                raise ValueError(f"Each item must contain '{IMAGE_PATH_KEY}'")

        try:
            if max_concurrency:
                config = {"max_concurrency": max_concurrency}
            else:
                config = None
            result = self.pipeline.batch(batch, config=config)
            logger.info("Batch processed successfully")
            return result
        except (OpenAIError, ValueError) as e:
            logger.error("Pipeline batch execution failed. Batch size: %d. Error: %s", len(batch), str(e),
                         exc_info=True)
            raise
        finally:
            duration = time.time() - start_time
            logger.info("Signature batch completed in %.2f seconds", duration)


class Signature2Brand:
    """Maps extracted text signatures to brand names and country codes.

    Uses LangChain pipelines combined with web search to determine which cosmetic or
    skincare brand a signature belongs to.
    """

    def __init__(self, model_name: str, web_search_service: WebSearchService):
        """Initializes the brand identification pipeline."""
        logger.info("Initializing Signature2Brand")

        # Error handling is in the function model_activation
        model = model_activation(model_name)

        output_parser, self.format_instructions = self._build_brand_parser()

        self.web_search_service = web_search_service  #
        websearch_runnable = RunnableLambda(self.web_search_service.search)
        # websearch_runnable = RunnableLambda(self.websearch)

        prompt_template = self.build_prompt_template()

        step_1 = RunnablePassthrough.assign(messages=self.build_websearch_message)
        step_2 = RunnablePassthrough.assign(websearch_text=itemgetter("messages")|websearch_runnable)
        step_3 = RunnablePassthrough.assign(brand=prompt_template|model|output_parser|self.extract_name_field)

        self.pipeline = build_pipeline([step_1, step_2, step_3])

    @staticmethod
    def _build_brand_parser() -> Tuple[PydanticOutputParser, str]:
        """Builds a parser for structured brand name output.

        Returns:
            Tuple[PydanticOutputParser, str]: Output parser and format instructions.
        """
        output_parser = PydanticOutputParser(pydantic_object=BrandOutput)
        format_instructions = output_parser.get_format_instructions()

        return output_parser, format_instructions

    @chain
    @staticmethod
    def extract_name_field(pydantic_object) -> Dict:

        return pydantic_object.model_dump()

    @chain
    @staticmethod
    def build_websearch_message(kwargs)-> List[Dict]:

        messages = [
                    {"role": "system", "content": dedent("""
                            You will find the cosmetic or skin care brand and the corresponding country 
                            of origin with a provided brand signature. Please search with the entire brand signature.
                            """)},
                    {"role": "user", "content": f"signature: {kwargs['signature']}?"}
                ]

        return messages

    def build_prompt_template(self) -> Runnable:
        """Constructs a chat prompt template to identify brands from web search text.

        Returns:
            Runnable: A chain-ready prompt object.
        """
        system_template = (
            "You are a helpful and detail-oriented AI assistant working as the personal assistant "
            "to a trend manager in the beauty and skincare industry.\n"
            "You will receive a paragraph obtained from web search based a "
            "signature of a cosmetic or skin care brand. You are going to help me identify the brand.\n"
            "If it is not about a brand but an influencer, the brand name is ``, an empty python string"
        )

        human_template = ("Paragraph: {websearch_text}\n"
                         "Output format instruction: {format_instructions}")

        input_ = {"system": {"template": system_template},
                  "human": {"template": human_template,
                            "input_variables": ['websearch_text'],
                            "partial_variables": {"format_instructions": self.format_instructions}}
                  }

        return build_standard_chat_prompt_template(input_)

    def batch(self, batch: List[Dict[str, str]], max_concurrency: Optional[int]=None) -> List[Dict[str, str]]:
        """Runs the brand identification pipeline on a batch of signature strings.

        Args:
            batch (List[Dict[str, str]]): List of items containing a 'signature'.
            max_concurrency (Optional[int]): Maximum number of concurrent model executions.

        Returns:
            List[Dict[str, str]]: List of items containing brand and country_code fields.

        Raises:
            ValueError: If 'signature' is missing in any batch item.
            OpenAIError: If model inference fails.
        """
        logger.info(f"Processing brand identification for {len(batch)} signatures")
        start_time = time.time()

        for item in batch:
            if SIGNATURE_KEY not in item:
                logger.warning("Missing required key: %s", SIGNATURE_KEY)
                raise ValueError(f"Each item must contain '{SIGNATURE_KEY}'")
        try:
            if max_concurrency:
                config = {"max_concurrency": max_concurrency}
            else:
                config = None
            result = self.pipeline.batch(batch, config=config)
            logger.info("Brand batch processing successful")
            for a in result:
                # decompress the brand into brand and country_code
                a.update(a['brand'])
            return result
        except (OpenAIError, ValueError) as e:
            logger.error("Brand pipeline failed for batch of size %d. Error: %s", len(batch), str(e), exc_info=True)
            raise
        finally:
            duration = time.time() - start_time
            logger.info("Brand batch completed in %.2f seconds", duration)