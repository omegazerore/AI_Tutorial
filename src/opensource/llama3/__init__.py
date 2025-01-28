import torch
from torch import cuda
from transformers import BitsAndBytesConfig


LLAMA3_8B_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else 'cpu'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

TEST_PROMPT = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are a honest and unbiased AI assistant who answer User queries with accurate responses.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            What's the capital of Australia?
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """