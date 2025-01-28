import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import chain, Runnable, RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document

from src.initialization import credential_init
from src.io.path_definition import get_project_dir


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


class TechPriest:

    def __init__(self, model: ChatOpenAI):

        self._build_chat_prompt_template()

        self._build_retriever()

        input_pipeline_ = RunnableParallel({"text": RunnablePassthrough(),
                                            "context":self.retriever|self.context_parser})
                           # ""}|RunnablePassthrough.assign(context=itemgetter("text")self.retriever|self.context_parser)

        self.pipeline_ = input_pipeline_|self.prompt_template|model|StrOutputParser()

    def _build_retriever(self):

        """
        Delete document in vectorstore

        vectorstore.docstore._dict.delele(ids=[...])
        """

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        path = os.path.join(get_project_dir(), 'tutorial', 'LLM+Langchain', 'Week-7', 'Codex - Adeptus Mechanicus Index')

        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

        self.retriever = vectorstore.as_retriever()

    @staticmethod
    @chain
    def context_parser(documents: List[Document]) -> str:
        context = ""

        for document in documents:
            context += f"{document.page_content}\n"

        return context

    def _build_chat_prompt_template(self):
        system_template = '''
        You are an AI assistant acting as a tech-priest of the Adeptus Mechanicus (Techsorcist). 
        
        You will answer the question based on the following information
        {context} 
        '''

        human_template = """
                         {text}\n
                         """

        input_ = {"system": {"template": system_template,
                             "input_variables": ['context']},
                  "human": {"template": human_template,
                            "input_variables": ['text']}}

        self.prompt_template = build_standard_chat_prompt_template(input_)


if __name__ == "__main__":

    from src.initialization import credential_init

    credential_init()

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini-2024-07-18", temperature=0)

    priest = TechPriest(model=model)

    question = "Tell me something about Belisarius Cawl."
    answer = priest.pipeline_.invoke(question)

    print(f"question: {question}\nanswer: {answer}")

    question = "What are the skills and weapons of Belisarius Cawl."
    answer = priest.pipeline_.invoke(question)

    print(f"question: {question}\nanswer: {answer}")