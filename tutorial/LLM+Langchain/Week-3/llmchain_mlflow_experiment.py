import os

import mlflow
from mlflow.models import set_model
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from src.initialization import credential_init


class LLMChainModel(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):

        # 固定套路: 把你的模型塞在這裡
        
        credential_init()

        model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                           model_name="gpt-4o-mini", temperature=0)
        
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )
        
        # Create the LLMChain with the specified model and prompt
        # 最早我也是用這個
        self.pipeline = LLMChain(llm=model, prompt=prompt)

    def predict(self, context, model_input):

        # model_input is a pandas dataframe

        # Give a marker 
        # print("\nHow are you today?\n")
        
        return self.pipeline.invoke({"product": model_input.loc[0]['input']})

set_model(LLMChainModel())
