from textwrap import dedent
from typing import Literal

import pandas as pd
from langchain.tools import BaseTool
from langchain.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import ConfigurableField
from langchain.docstore.document import Document
from pydantic import BaseModel, Field


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vectorstore = FAISS.load_local(
    "tutorial/LLM+Langchain/Week-5/warhammer 40k codex", embeddings, 
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_type="similarity", k=10).configurable_fields( \
                                        search_kwargs=ConfigurableField(
                                                id="search_kwargs",
                                            )
                                        )

class Inputs(BaseModel):
    query: str = Field(description="User query")
    clan: Literal['Adeptus Mechanicus', 'Aeldari', 'Black Templars'] = Field(description="")


class CodexRetrievalTool(BaseTool):

    input_output_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Inputs)
    input_format_instruction: str = input_output_parser.get_format_instructions()
    
    name:str = "Vectorestore tool for warhammer 40k codex"
    description_template:str = dedent("""
    This vectorstore tool can be used to retrieve relevant information about warhammer 40k, 
    particularly Adeptus Mechanicus (機械修會), Aeldari(艾達零族), Black Templars(黑色聖堂).
    When the user is asking questions about this three parties, this tool has a high priority than the websearch.
    The inputs contains user's question `query` and the party/clan `clan`.
    If you cannot find the answer from the vector, please use the websearch tool.
    
    input format instructions: {input_format_instruction}
    """)

    description: str = description_template.format(input_format_instruction=input_format_instruction)
    
    def _run(self, query):
        
        input_ = self.input_output_parser.parse(query)

        query = input_.query
        clan = input_.clan
        
        if clan == 'Black Templars':
            filter_ = {"filename": f"Codex -{clan}"}
        else:
            filter_ = {"filename": f"Codex - {clan}"}
        
        retrieved_documents = retriever.invoke(query, config={"configurable": 
                                                             {"search_kwarg": {"filter": filter_
                                                                              }
                                                             }
                                                            }
                                                     )

        context = "\n\n".join([document.page_content for document in retrieved_documents])
        
        return context
    
    async def _arun(self, query: str):
        
        input_ = self.input_output_parser.parse(query)

        query = input_.query
        clan = input_.clan
        
        if clan == 'Black Templars':
            filter_ = {"filename": f"Codex -{clan}"}
        else:
            filter_ = {"filename": f"Codex - {clan}"}
        
        retrieved_documents = await retriever.ainvoke(query, config={"configurable": 
                                                             {"search_kwarg": {"filter": filter_
                                                                              }
                                                             }
                                                            }
                                             )

        context = "\n\n".join([document.page_content for document in retrieved_documents])
        
        return context