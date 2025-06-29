import os
from typing import List, Dict, Optional

from openai import OpenAI

from src.initialization import credential_init


class WebSearchService:
    """Service class responsible for querying OpenAI's web search endpoint."""

    def __init__(self, client: OpenAI, search_context_size: str, model: str):

        self.client = client
        self.search_context_size = search_context_size
        self.model = model

    # def search(self, signature: str) -> str:
    #     """Searches for information related to a brand signature.
    #
    #     Args:
    #         signature (str): The brand signature string.
    #
    #     Returns:
    #         str: Search result text from OpenAI.
    #     """
    #     response = self.client.chat.completions.create(
    #         model=MODEL,
    #         web_search_options={"search_context_size": SEARCH_CONTEXT_SIZE},
    #         messages=[
    #             {"role": "system", "content": "You will find the brand with a provided signature."},
    #             {"role": "user", "content": f"signature: {signature}?"}
    #         ]
    #     )
    #     return response.choices[0].message.content
    def search(self, messages: List[Dict], country_code: Optional[str]=None) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            web_search_options={"search_context_size": self.search_context_size,
                                "user_location": {
                                        "type": "approximate",
                                        "approximate": {
                                            "country": country_code,
                                        }
                                    },
                                },
            messages=messages
        )
        return response.choices[0].message.content


def activate_websearch_service(model_name: str, search_context_size: str) -> WebSearchService:

    credential_init()
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")
    client = OpenAI(api_key=openai_key)

    websearch_service = WebSearchService(client=client, search_context_size=search_context_size,
                                         model=model_name)

    return websearch_service