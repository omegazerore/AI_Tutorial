import os

import mlflow
import mlflow.pyfunc
from mlflow.models import set_model
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from openai import OpenAI

from src.initialization import credential_init
from src.agent.react_zero_shot import prompt_template as zero_shot_prompt_template


credential_init()


class MyConsoleCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        # print(f"LLM started with prompts: {prompts}")
        mlflow.log_dict({"prompts": prompts}, "traces/llm_prompts.json",
                        run_id=os.environ["MLFLOW_RUN_ID"])
    def on_llm_end(self, response, **kwargs):
        # print(f"LLM responded: {response}")
        outputs = [gen.text for gen in response.generations[0]]
        mlflow.log_dict({"responses": outputs}, "traces/llm_responses.json",
                        run_id=os.environ["MLFLOW_RUN_ID"])

@chain
def gpt_web_search_tool(text):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={"search_context_size": "medium"},
        messages=[{"role": "user",
                   "content": text}]
    )

    return response.choices[0].message.content


class SearchTool(BaseTool):
    name: str = "Search Engine"
    description: str = "Use this tool to find the knowledge you need."

    def _run(self, query: str):
        response = gpt_web_search_tool.invoke(query)

        return response

    async def _arun(self, query: str):
        response = await gpt_web_search_tool.ainvoke(query)

        return response


class AgentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # rebuild your agent here
        prompt = PromptTemplate(template=zero_shot_prompt_template)

        tools = [SearchTool()]

        llm_gpt_4o_mini = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                                     model_name="gpt-4o-mini", temperature=0, callbacks=[MyConsoleCallback()])

        zero_shot_agent = create_react_agent(
            llm=llm_gpt_4o_mini,
            tools=tools,
            prompt=prompt,
        )
        self.agent_executor = AgentExecutor(agent=zero_shot_agent, tools=tools, verbose=True,
                                            handle_parsing_errors=True)

    def predict(self, context, model_input):
        return self.agent_executor.invoke({"input": model_input["input"]})


set_model(AgentModel())
