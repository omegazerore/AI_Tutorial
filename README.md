# AICG_Tutorial

## Prerequisites

### External API
    
- OPENAI API
- SERPER API: https://serper.dev/?gad_source=1&gclid=Cj0KCQjw-ai0BhDPARIsAB6hmP5H8rzdauXgelHyXlYimy-AB4Qu2xfn-VkLudzakgIuud4dId-_m-waAhYSEALw_wcB
- TAVILY API: https://tavily.com/

### Account:

- HuggingFace
- Goggle Colab
- LLAMA2 & LLAMA3 Permission Application

## Conda Environment Installation

We will setup two environments, one for agent, one for non-agent.

In the Anacondo Prompt:

Agent Environment:
    
    conda create -n llm_agent python=3.10 -y
    conda activate llm_agent
    pip install -r requirements_agent.txt


Non-Agent Environment:

    conda create -n llm python=3.10 -y
    conda activate llm
    pip install -r requirements.txt