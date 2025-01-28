"""
This is an example of how to apply Llama3 on a translator
"""


import os
from operator import itemgetter
from typing import List

import pandas as pd
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, chain, Runnable
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.opensource.llama3.llama3_8b import build_llm, llama3_prompt_parser


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

    def __init__(self, model_id: str):

        self._model_id = model_id
        self.model = build_llm(model_id=model_id)
        self._build_retriever()
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
                           description="language of the text. If receiving 'I do not know' as the answer, "
                                       "the answer is <UNKNOWN>")
        ]

        # Create an output parser based on the response schemas
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # Get format instruction  ons from the output parser
        format_instructions = output_parser.get_format_instructions()

        system_template = """
                          You are a helpful AI assistant as a linguistic expert.
                          """

        human_template = """
                         What is the corresponding language? \n\n {text}\n{format_instructions}
                         """

        input_ = {"system": {"template": system_template},
                  "human": {"template": human_template,
                            "input_variables": ['text'],
                            "partial_variables": {"format_instructions": format_instructions}
                            }
                  }

        chat_prompt_template = build_standard_chat_prompt_template(input_)

        pipeline_ = chat_prompt_template | self.model | output_parser | itemgetter('language')

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
                          You are a helpful AI assistant as a linguistic expert.
                          
                          You will translate text from language <{language}> to language {target_language}.
                          
                          You will follow the translation style shown by <context> provided by user. 
                          
                          The output will follow the format instruction: {format_instructions}
                          """

        human_template = """
                         text: {text}
                         
                         context: \n
                         {context}
                         """

        input_ = {"system": {"template": system_template,
                             "input_variables": ['language', 'target_language'],
                             "partial_variables": {"format_instructions": format_instructions}
                             },
                  "human": {"template": human_template,
                            "input_variables": ['text', 'context']
                            }
                  }

        chat_prompt_template = build_standard_chat_prompt_template(input_)

        step_1 = RunnablePassthrough.assign(context=itemgetter('text')|self.retriever|self.context_parser,
                                            language=self._identification_pipeline)|chat_prompt_template

        pipeline_ = step_1 | llama3_prompt_parser| self.model | output_parser | itemgetter('translation')  # wrapper# wrapper

        return pipeline_

    def _build_retriever(self):

        from src.io.path_definition import get_project_dir

        filename = os.path.join(get_project_dir(), "translation_pairs_en_de.csv")

        df = pd.read_csv(filename)

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        documents = self._build_documents(df)

        vectorstore = FAISS.from_documents(documents, embedding=embedding)

        retriever = vectorstore.as_retriever(search_type='similarity',
                                             search_kwargs={'k': 5})

        self.retriever = retriever


    def _build_documents(self, df: pd.DataFrame):

        documents = []

        for idx, row in df.iterrows():
            document = Document(page_content=row['source_text'],
                                metadata={"target text": row["target_text"]})
            documents.append(document)

        return documents

    @staticmethod
    @chain
    def context_parser(documents: List[Document]) -> str:

        context = ""

        for document in documents:
            context += f"user: {document.page_content}\t"
            context += f"assistant: {document.metadata['target text']}\n"

        return context
