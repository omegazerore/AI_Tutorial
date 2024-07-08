import logging
import sys
import os

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain.chat_models import ChatOpenAI

from src.initialization import credential_init

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

case = "tutorial_example"


credential_init()


model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o-2024-05-13", temperature=0)

app = FastAPI(
    title="hr assistant",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    model,
    path="/openai",
)

if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=5000)