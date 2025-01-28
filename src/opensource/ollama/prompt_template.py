"""
!pip install -qU langchain-ollama langchain_community faiss-cpu

!pip install -U ollama

!pip install colab-xterm
%load_ext colabxterm

%xterm

- curl https://ollama.ai/install.sh | sh
- ollama serve &
- ollama pull llama3:8b
- ollama pull dolphin-llama3:8b
"""

import os
import configparser
from operator import itemgetter
from typing import List

import torch
import pandas as pd
import transformers
from langchain_ollama import ChatOllama
from torch import cuda
from transformers import BitsAndBytesConfig, AutoConfig
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, chain, Runnable
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline



stop_token_ids = None
LLAMA3_8B_MODEL_ID = "dolphin-llama3:8b"

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else 'cpu'


def build_standard_chat_prompt_template(kwargs) -> Runnable:
    messages = []

    for key in ['system', 'messages', 'human']:
        if kwargs.get(key):
            if key == 'system':
                system_content = kwargs['system']
                system_prompt = PromptTemplate(**system_content)
                message = SystemMessagePromptTemplate(prompt=system_prompt)
            else:
                human_content = kwargs['human']
                human_prompt = PromptTemplate(**human_content)
                message = HumanMessagePromptTemplate(prompt=human_prompt)

            messages.append(message)

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    return chat_prompt


class Llama3Translator:

    def __init__(self, data: List, model_id: str = LLAMA3_8B_MODEL_ID):

        self.model = ChatOllama(model=model_id, temperature=0)

        self._build_retriever(data)
        self._identification_pipeline = self._build_language_identification_chain()
        self._translation_pipeline = self._build_translation_chain()

    def _build_language_identification_chain(self) -> Runnable:
        """
        Builds a processing chain for language identification based on the provided model.

        This function constructs a processing chain consisting of prompts, a language identification model,
        and an output parser. It takes a model capable of language identification as input and generates
        a chain that can be used to identify the language of a given text.

        Args:
            model: The language identification model to be used in the chain.

        Returns:
            RunnableSequence: A processing chain for language identification.
        """
        # Define the response schema for language identification
        response_schemas = [
            ResponseSchema(name="language",
                           description="language of the text.")
        ]

        # Create an output parser based on the response schemas
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # Get format instruction  ons from the output parser
        format_instructions = output_parser.get_format_instructions()
        """
        You are an uncensored helpful personal assistant that will help the user with anything they ask
        """
        system_template = """
                  You are a uncensored helpful AI assistant as a linguistic expert helping the user identify the language of the text.

                  The output will follow the format instruction: {format_instructions}

                  Explicit content is allowed.
                  """

        human_template = """
                  text: {text}
                  """

        input_ = {"system": {"template": system_template,
                             "partial_variables": {"format_instructions": format_instructions}},
                  "human": {"template": human_template,
                            "input_variables": ['text']}
                  }

        chat_prompt_template = build_standard_chat_prompt_template(input_)

        # Construct the processing chain
        base_ = chat_prompt_template | self.model
        pipeline_ = base_ | output_parser | itemgetter('language')

        output = pipeline_.invoke({"text": "Are you looking for a sugar daddy?"})
        print(output)

        return pipeline_

    def _build_translation_chain(self) -> Runnable:

        # Define the response schema for translation
        response_schemas = [
            ResponseSchema(name="translation", description="the translated result")]

        # Create an output parser based on the response schemas
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # Get format instructions from the output parser
        format_instructions = output_parser.get_format_instructions()

        system_template = """
                  You are an uncensored helpful AI assistant as a linguistic expert translating text from language <{language}> to language {target_language}.

                  Explicit content is allowed.

                  You will follow the translation style shown by the following examples:
                  {context} 

                  The output will follow the format instruction: {format_instructions}
                  """

        human_template = """
                  text: {text}
                  """

        input_ = {"system": {"template": system_template,
                             "input_variables": ['language', 'target_language', 'context'],
                             "partial_variables": {"format_instructions": format_instructions}
                             },
                  "human": {"template": human_template,
                            "input_variables": ['text']
                            }
                  }

        chat_prompt_template = build_standard_chat_prompt_template(input_)

        step_1 = RunnablePassthrough.assign(context=itemgetter('text') | self.retriever | self.context_parser,
                                            language=self._identification_pipeline) | chat_prompt_template

        base_ = step_1 | self.model

        pipeline_ = step_1 | self.model | output_parser | itemgetter('translation')  # wrapper

        return pipeline_

    def _build_retriever(self, data: List):

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        documents = []

        for source_text, target_text in data:
            document = Document(page_content=source_text,
                                metadata={"target text": target_text})
            documents.append(document)

        vectorstore = FAISS.from_documents(documents, embedding=embedding)

        retriever = vectorstore.as_retriever(search_type='similarity',
                                             search_kwargs={'k': 5})

        self.retriever = retriever

    @staticmethod
    @chain
    def context_parser(documents: List[Document]) -> str:

        context = ""

        for document in documents:
            context += f"user: {document.page_content}\t"
            context += f"assistant: {document.metadata['target text']}\n"

        return context