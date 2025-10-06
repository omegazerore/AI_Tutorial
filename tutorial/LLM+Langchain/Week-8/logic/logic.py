"""
image_generation_pipeline.py


This module provides utility functions and pipelines for generating, editing,
and saving images using OpenAI's GPT-based image models combined with LangChain
prompt orchestration. It enables natural language-driven image creation and
modification, supporting multimodal prompts and seamless integration with
LangChain Runnables.


Dependencies:
- openai: For GPT-based image generation and editing.
- langchain, langchain_openai, langchain_core: For prompt templates,
message orchestration, and runnable chains.
- base64, io, os: For image encoding/decoding and file handling.


Environment:
Requires the `OPENAI_API_KEY` environment variable to be set or initialized
through `credential_init()` from `src.initialization`.


Example:
>>> from image_generation_pipeline import image_pipeline
>>> chain = image_pipeline("You are a helpful assistant that generates illustrations.")
>>> result = chain.invoke({"story": "A serene lake under a starry night sky"})
>>> with open("lake.png", "wb") as f:
... f.write(result.getvalue())
"""

import base64
import io
import os
from operator import itemgetter
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts.image import ImagePromptTemplate
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain, RunnableLambda, RunnableParallel, RunnablePassthrough
from openai import OpenAI

from src.initialization import credential_init

credential_init()
# os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY"

client = OpenAI()


def build_standard_chat_prompt_template(kwargs):
    """Builds a standardized LangChain ChatPromptTemplate from input prompts.


    Supports both system and human prompts, including multimodal (text and image)
    configurations.
    
    
    Args:
    kwargs (dict): Dictionary containing optional `system` and `human` keys.
    - system (dict or list[dict]): System prompt configuration(s).
    - human (dict or list[dict]): Human prompt configuration(s).
    
    
    Returns:
    ChatPromptTemplate: A composed chat prompt template containing system and
    human message prompts.
    """
    
    messages = []

    if 'system' in kwargs:
        content = kwargs.get('system')

        # allow list of prompts for multimodal
        if isinstance(content, list):
            prompts = [PromptTemplate(**c) for c in content]
        else:
            prompts = [PromptTemplate(**content)]

        message = SystemMessagePromptTemplate(prompt=prompts)
        messages.append(message)

    if 'human' in kwargs:
        content = kwargs.get('human')

        # allow list of prompts for multimodal
        if isinstance(content, list):
            prompts = []
            for c in content:
                if c.get("type") == "image":
                    prompts.append(ImagePromptTemplate(**c))
                else:
                    prompts.append(PromptTemplate(**c))
        else:
            if content.get("type") == "image":
                prompts = [ImagePromptTemplate(**content)]
            else:
                prompts = [PromptTemplate(**content)]

        message = HumanMessagePromptTemplate(prompt=prompts)
        messages.append(message)

    chat_prompt_template = ChatPromptTemplate.from_messages(messages)
    
    return chat_prompt_template


@chain
def gpt_image_worker(kwargs: Dict) -> str:

    """Generates an image from a natural language prompt using OpenAI's GPT Image API.


    Args:
    kwargs (dict): Dictionary with keys:
    - nl_prompt (str): Natural language description of the image.
    - size (str, optional): Image resolution (default: "1024x1024").
    - quality (str, optional): Image quality (default: "medium").
    - moderation (str, optional): Moderation mode (default: "auto").
    
    
    Returns:
    str: Base64-encoded image string.
    """
    
    response = client.images.generate(
        model="gpt-image-1",
        prompt=kwargs['nl_prompt'],
        size=kwargs.get("size", "1024x1024"),
        quality=kwargs.get('quality', 'medium'),
        moderation=kwargs.get('moderation', 'auto'),
        n=1)

    image_base64 = response.data[0].b64_json
    
    return image_base64


@chain
def gpt_image_render(kwargs) -> str:

    """Edits an existing image using OpenAI's GPT Image API.


    Args:
    kwargs (dict): Dictionary with keys:
    - nl_prompt (str): Instructions for editing the image.
    - image_io (BytesIO): Input image file-like object.
    - size (str, optional): Output image resolution (default: "1024x1024").
    - quality (str, optional): Output image quality (default: "medium").
    
    
    Returns:
    str: Base64-encoded image string after editing.
    """
    
    response = client.images.edit(
        model="gpt-image-1",
        image=[kwargs['image_io']],
        prompt=kwargs['nl_prompt'],
        size=kwargs.get("size", "1024x1024"),
        quality=kwargs.get('quality', 'medium'),
        n=1)

    image_base64 = response.data[0].b64_json
    
    return image_base64


@chain
def base64_to_file(kwargs) -> io.BytesIO:

    """Decodes a base64 image string and saves it to a file.


    Args:
    kwargs (dict): Dictionary with keys:
    - image_base64 (str): Base64-encoded image string.
    - filename (str): Output file path.
    
    
    Returns:
    io.BytesIO: In-memory file object containing the image.
    """
    
    image_base64 = kwargs['image_base64']

    # Decode to bytes
    image_bytes = base64.b64decode(image_base64)
    
    with open(kwargs['filename'], "wb") as fh:
        fh.write(image_bytes)

    # # Wrap in a BytesIO object
    image_file = io.BytesIO(image_bytes)
    image_file.name = kwargs['filename']
    
    # Wrap BytesIO in a BufferedReader (no extra disk read)
    # image_file = io.BufferedReader(io.BytesIO(image_bytes))
    
    return image_file


def image_pipeline(system_template: str):

    """Creates a pipeline for generating new images from text descriptions.


    Args:
    system_template (str): System-level instruction for the model.
    
    
    Returns:
    Runnable: A chain that:
    1. Generates a natural language prompt from story input.
    2. Produces an image via `gpt_image_worker`.
    3. Saves the image to disk with `base64_to_file`.
    """
    
    input_ = {"system": {"template": system_template},
              "human": {"template": "{story}",
                        "input_variable": ["story"]}}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini", temperature=0)
    
    nl_prompt_generation_chain = chat_prompt_template | model | StrOutputParser()     
    
    step_1 = RunnablePassthrough.assign(nl_prompt=itemgetter('story')|nl_prompt_generation_chain)
    step_2 = RunnablePassthrough.assign(image_base64=gpt_image_worker)
    step_3 = base64_to_file
    image_chain = step_1 | step_2 | step_3

    return image_chain


def image_render_pipeline(system_template: str):

    """Creates a pipeline for editing images based on text instructions.


    Args:
    system_template (str): System-level instruction for the model.
    
    
    Returns:
    Runnable: A chain that:
    1. Generates a natural language prompt from story input.
    2. Edits an existing image via `gpt_image_render`.
    3. Saves the modified image to disk with `base64_to_file`.
"""
    
    input_ = {"system": {"template": system_template},
              "human": {"template": "{story}",
                        "input_variable": ["story"]}}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini", temperature=0)
    
    nl_prompt_generation_chain = chat_prompt_template | model | StrOutputParser()     
    
    step_1 = RunnablePassthrough.assign(nl_prompt=itemgetter('story')|nl_prompt_generation_chain)
    step_2 = RunnablePassthrough.assign(image_base64=gpt_image_render)
    step_3 = base64_to_file
    image_chain = step_1 | step_2 | step_3

    return image_chain


def story_pipeline(system_template: str):

    """Creates a pipeline for generating stories or narratives.


    Args:
    system_template (str): System-level instruction for the model.
    
    
    Returns:
    Runnable: A chain that generates a text story from input using GPT-4o-mini.
    """
    
    system_prompt = PromptTemplate(template=system_template)

    input_ = {"system": {"template": system_template},
              "human": {"template": "scratch: {scratch}\nwhat happens previously: {context}",
                        "input_variable": ["scratch", "context"]}}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini", temperature=0)
    
    story_chain = chat_prompt_template | model | StrOutputParser()

    return story_chain