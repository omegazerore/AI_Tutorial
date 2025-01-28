import os

import transformers
from transformers import AutoConfig
from langchain.llms import HuggingFacePipeline
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

from src.initialization import credential_init
from src.opensource.llama3 import LLAMA3_8B_MODEL_ID, bnb_config, TEST_PROMPT


def load_model(model_id: str = LLAMA3_8B_MODEL_ID):

    credential_init()

    HF_TOKEN = os.environ['HuggingFace_API_KEY']

    model_config = AutoConfig.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        token=HF_TOKEN,
        config=model_config,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)

    return model, tokenizer


def build_llm(temperature: float=0.5, max_new_tokens: int=512,
              repetition_penalty: float=1.2, top_p=0.5, model_id: str=LLAMA3_8B_MODEL_ID):

    if model_id not in ['openai']:
        model, tokenizer = load_model(model_id)

        terminators = [
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        text_generation_pipeline = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=False,
            task='text-generation',
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            do_sample=True,
            eos_token_id=terminators,
            max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
            repetition_penalty=repetition_penalty,  # without this output begins repeating
            top_p=top_p,
            pad_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"))


        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        print("\n***********************\n")
        print(llm.invoke(TEST_PROMPT))
        print("\n***********************\n")
    else:

        credential_init()

        llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                           model_name="gpt-4o-mini", temperature=0)

    return llm


@chain
def llama3_prompt_parser(prompt):

    prompt_template = """\n<|begin_of_text|>"""

    for message in prompt.messages:
        if message.type == "system":
            prompt_template += f"\n<|start_header_id|>system<|end_header_id|>\n{message.content}\n<|eot_id|>"
        elif message.type == "human":
            prompt_template += f"\n<|start_header_id|>user<|end_header_id|>{message.content}\n<|eot_id|>"

    prompt_template += f"\n<|start_header_id|>assistant<|end_header_id|>\n"

    return prompt_template

