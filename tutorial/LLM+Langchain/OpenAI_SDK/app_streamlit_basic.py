import asyncio
import requests

import streamlit as st
from agents import Agent, Runner, SQLiteSession
from agents.extensions.models.litellm_model import LitellmModel
from litellm.exceptions import InternalServerError
from openai.types.responses import ResponseTextDeltaEvent

from src.initialization import credential_init

import os

# Initialize ChatMessageHistory in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session" not in st.session_state:
    st.session_state.session = SQLiteSession("session")

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "user_input_temp" not in st.session_state:
    st.session_state.user_input_temp = ""

if "send_on_enter" not in st.session_state:
    st.session_state.send_on_enter = False

if "agent" not in st.session_state:

    credential_init()
    
    model = LitellmModel(model="gemini/gemini-2.0-flash", api_key=os.environ["GOOGLE_API_KEY"])
    
    st.session_state.agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
        model=model)


def submit():
    """
    Clear the input after submit
    """
    st.session_state.user_input_temp = st.session_state.user_input
    st.session_state.user_input = ''            # clear the visible input
    if st.session_state.user_input_temp != '':
        st.session_state.send_on_enter = True       # signal main loop to send
    else:
        st.session_state.send_on_enter = False

async def run():

    user_input = st.session_state.user_input_temp
    
    if user_input.strip() != "":
        result = await Runner.run(
            st.session_state.agent,
            user_input,
            session=st.session_state.chat_history
        )


async def run_streaming():
    user_input = st.session_state.user_input_temp

    if not user_input.strip():
        return

    result = Runner.run_streamed(st.session_state.agent,
                                 input=user_input,
                                 session=st.session_state.session)

    st.session_state.chat_history.append(("user", user_input))

    placeholder = st.empty()
    chat_messages = ""
    if len(st.session_state.chat_history) > 0:
        for msg in st.session_state.chat_history:
            role = msg[0]
            if role == "user":
                chat_messages += f"<div class='user-msg'>🧑 <b>You:</b> {msg[1]}</div>"
            else:
                chat_messages += f"<div class='ai-msg'>🤖 <b>AI:</b> {msg[1]}</div>"

    print(chat_messages)

    ai_message = ""
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            ai_message += event.data.delta
            chat_messages_partial = chat_messages + f"<div class='ai-msg'>🤖 <b>AI:</b> {ai_message}</div>"
            placeholder.markdown(
                f"<div class='chat-box'>{chat_messages_partial}</div>",
                unsafe_allow_html=True
            )

    st.session_state.chat_history.append(("ai", ai_message))


# ---- UI ----
st.set_page_config(page_title="OpenAI Agent 聊天機器人 Demo", layout="wide")

st.title("💬 Chat with Conversational Agent")

# --- CSS for Slack-like scrollable chat window ---
st.markdown("""
    <style>
    .chat-box {
        height: 500px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 10px;
        background-color: #fafafa;
    }

    /* --- user message (靠右對齊) --- */
    .user-msg {
        color: #1d1d1d;
        font-weight: 600;
        margin-bottom: 8px;
        text-align: right;
        background-color: #dcf8c6;   /* 淺綠色背景像聊天氣泡 */
        padding: 8px 12px;
        border-radius: 15px 15px 0 15px;
        display: inline-block;
        max-width: 80%;
        align-self: flex-end;
        float: right;
        clear: both;
    }

    /* --- ai message (靠左對齊) --- */
    .ai-msg {
        color: #0a66c2;
        margin-bottom: 8px;
        text-align: left;
        background-color: #ffffff;
        padding: 8px 12px;
        border-radius: 15px 15px 15px 0;
        display: inline-block;
        max-width: 80%;
        float: left;
        clear: both;
    }
    </style>
""", unsafe_allow_html=True)

st.header("Chat History")
chat_container = st.container()
input_container = st.container()

with input_container:
    _ = st.text_input(
        "Type your message...", 
        key="user_input", 
        on_change=submit,
    )

    col1, col2 = st.columns([8, 1])
    with col2:
        send_button = st.button("Send")
    
if send_button or st.session_state.send_on_enter:
    with chat_container:
        asyncio.run(run_streaming())
