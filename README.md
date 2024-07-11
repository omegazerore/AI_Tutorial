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
    conda install -c pytorch faiss-cpu    

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

### Week 3

- 利用語言模型實現高效文字分類
  - 1.數據預處理技術與工具介紹
  - 2.零樣本學習/分類
  - 3.零樣本學習的基本概念與應用場景
  - 4.介紹N樣本學習的概念與應用

### Week 4

- 打造互動式聊天機器人與遠端服務部屬
  - 1.LangChain Client/Server 功能實作(遠端服務部屬、客戶端服務取得)
  - 2.Streaming技術概述
  - 3.聊天機器人的基本概念與應用
