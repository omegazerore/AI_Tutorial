from langchain_core.runnables import Runnable
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate


def build_standard_chat_prompt_template(kwargs) -> Runnable:
    messages = []

    for key in ['system', 'messages', 'human']:
        if kwargs.get(key):
            if key == 'system':
                system_content = kwargs['system']
                system_prompt = PromptTemplate(**system_content)
                message = SystemMessagePromptTemplate(prompt=system_prompt)
            else:
                human_content = kwargs['human']
                human_prompt = PromptTemplate(**human_content)
                message = HumanMessagePromptTemplate(prompt=human_prompt)

            messages.append(message)

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    return chat_prompt