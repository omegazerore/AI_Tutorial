import logging
import sys
import os
from operator import itemgetter

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.messages import AIMessage


from src.initialization import credential_init
from src.trendweek_report.deep_search import OpenAIWebSearch
from src.trendweek_report.examples_with_deep_search import ExampleSearch, response_to_output_text

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o-2024-05-13", temperature=0)

app = FastAPI(
    title="tutorial",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    model,
    path="/openai",
)

############################################################################################

@chain
def final_output_2_ai_message(input_):

    return AIMessage(content=input_['final_output']['result'])


example_search = ExampleSearch(model='gpt-4o-mini')
openai_websearch = OpenAIWebSearch(model="gpt-4o", temperature=0)

pipeline_ = {'question': example_search.prompt_pipeline}|openai_websearch.pipeline|RunnablePassthrough.assign(final_output=itemgetter("context")|response_to_output_text|example_search.parsing_pipeline)|final_output_2_ai_message

add_routes(
    app,
    pipeline_,
    path="/openai_websearch",
)

if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=5000)