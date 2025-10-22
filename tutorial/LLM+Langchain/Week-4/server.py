import logging
import sys
import os
from typing import Dict, List
from operator import itemgetter

import mlflow
import uvicorn
import pandas as pd
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.messages import AIMessage


from src.initialization import credential_init
# from src.trendweek_report.deep_search import OpenAIWebSearch
# from src.trendweek_report.examples_with_deep_search import SearchQueryAssistant, response_to_output_text

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o-mini", temperature=0)

app = FastAPI(
    title="tutorial",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# LangChain 的 LangServe 提供的工具，用來把一個 Runnable 模型掛載到 API 端點上。
add_routes(
    app,
    model,
    path="/openai",
)

# 掛載後，會自動生成 /openai/invoke, /openai/stream, /openai/batch 等 API endpoint。

# 下載模型

run_name = "Reflection"
experiment = "Week-4"

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# Set the experiment
mlflow.set_experiment(experiment)

# Start the run and keep it open
run = mlflow.start_run(run_name=run_name)

os.environ['experiment'] = experiment
os.environ['run_id'] = run.info.run_id
os.environ['run_name'] = run_name

loaded_model = mlflow.pyfunc.load_model("models:/Generation_Reflection_Demo/1")

@chain
def final_output_2_ai_message(input_):

    # A wrapper which transform the output into an AIMessage object
    
    return AIMessage(content=input_)


@chain
def call_service(input_):

    df_input_ = pd.DataFrame(data=[input_])
    
    output = loaded_model.predict(df_input_)

    return output


pipeline = call_service|final_output_2_ai_message

add_routes(
    app,
    pipeline,
    path="/demo",
)


# IMPORTANT: Do not close the run here. Close it when server shuts down.
# For example, register a shutdown hook:
import atexit
atexit.register(mlflow.end_run)


############################################################################################




# example_search = SearchQueryAssistant(model='gpt-4o-mini')

# pipeline_ = {'question': example_search.prompt_pipeline}|openai_websearch.pipeline|RunnablePassthrough.assign(final_output=itemgetter("context")|response_to_output_text|example_search.parsing_pipeline)|final_output_2_ai_message

# add_routes(
#     app,
#     pipeline_,
#     path="/openai_websearch",
# )

if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=5000)