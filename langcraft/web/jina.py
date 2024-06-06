import json
import requests
from typing import List, Dict
from langcraft.action import Actions


def search_web_jina(query: str, max_results:int = 3) -> List[Dict]:
    """
    Search the web using the Jina AI search engine.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return, defaults to 3, maximum is 5.

    Returns:
        str: The response content if the request is successful.
    """
    response = requests.get(
        f"https://s.jina.ai/{query}", headers={"Accept": "application/json"}
    )

    if response.status_code == 200:
        return json.loads(response.content)["data"][:max_results]

    raise Exception(f"Error searching the web: {response.status_code}")


Actions.generate_action(search_web_jina)
