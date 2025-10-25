import importlib
import os
from textwrap import dedent
from typing import List
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables import chain, Runnable, RunnablePassthrough
from langchain_core.messages import BaseMessage

from src.initialization import credential_init

# from tutorial.LLM+Langchain.Week-7.websearch import SearchTool

#嘗試單純的加入聊天紀錄

template = dedent("""\
Answer the following questions as best you can. You have access to the following tools:

{tools}

When you need to use a tool to get knowledge, the vectorstore tools have a high priority than websearch tool.

To use a tool, you MUST strictly follow this format (case-sensitive, exact words)::

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: [the result of the action]

```

... (this Thought/Action/Action Input/Observation can repeat N times)


When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]. The final response should be in Traditional Chinese (繁體中文).

```

Begin!

Previous conversation history:

{chat_history}

Question: {input}

Thought:{agent_scratchpad}
"""
)

module_math = importlib.import_module("tutorial.LLM+Langchain.Week-7.tools.math")
module_vectorstore = importlib.import_module("tutorial.LLM+Langchain.Week-7.tools.vectorstore")
module_websearch = importlib.import_module("tutorial.LLM+Langchain.Week-7.tools.websearch")

tools = [module_math.MathTool(), module_websearch.SearchTool(), module_vectorstore.CodexRetrievalTool()]

prompt = PromptTemplate.from_template(template)

credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o", temperature=0, 
                  )

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    """
    Field:
     - 第一個參數 ... 代表 這個欄位是必填的。等同於 required=True。
    """
    input: str
    chat_history: List[BaseMessage] = Field(
        ...
    )


class Output(BaseModel):
    output: str
    input: str


conversation_agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=conversation_agent, tools=tools, verbose=True,
                               handle_parsing_errors=True).with_types(input_type=Input, output_type=Output)

# pipeline = debug#|agent_executor

app = FastAPI(title="conversational ReAct agent chatbot",
              version="1.0",
              description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    agent_executor,
    # pipeline,
    path="/chatbot",
)



if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=8080)