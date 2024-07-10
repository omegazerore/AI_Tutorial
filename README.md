# AICG_Tutorial

## Prerequisites

### External API

(*** They did not pay me ***)
    
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

## Syllabus

### Week 1

- 利用LangChain框架與OpenAI API提升指令下達與內容生成效率
  - 1.提示工程基本概念說明
  - 2.LangChain框架概念和功能介紹
  - 3.Outputparser作用和使用方法
  - 4.Okapi25數據檢索

### Week 2

- 掌握LangChain Expression Language (LCEL)：LLM開發必備技能
  - 1.LangChain 框架進行 embedding數據檢索
  - 2.LCEL概念與工作流程介紹
  - 3.LCEL語法結構與邏輯運算
  - 4.LCEL範例操作 
