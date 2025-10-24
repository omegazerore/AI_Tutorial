import io
import json
import os
import requests

from flask import Flask, request, Response
from langchain.memory import ChatMessageHistory

chat_history = ChatMessageHistory()

app = Flask(__name__)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get raw binary data from request body
    data = request.get_json()

    # Deserialize
    chat_history_ui = data.get("chat_history")

    print(f"\nchat_history_ui={chat_history_ui}\n")
    
    for message in chat_history_ui:
        if message['type'] == 'human':
            chat_history.add_user_message(message['content'])
        if message['type'] == 'ai':
            chat_history.add_ai_message(message['content'])
    
    response = requests.post(
        "http://localhost:8080/chatbot/invoke",
        json={'input': {"input": data['input'],
                        "chat_history" : [c.model_dump() for c in chat_history.messages]}}
    )

    response.encoding = 'utf-8'
    
    return response.json()['output']


if __name__ == "__main__":
    app.run(debug=True)