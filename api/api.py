import os
import httpx
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# Get the base URL from the environment variable
base_api_url = os.getenv('BASE_API_URL')

async def call_api(node_name: str, payload: Dict) -> Dict:
    """
    Function to call an API with a given URL, payload, and node name.

    Args:
        node_name (str): The name of the node to include in the URL.
        payload (Dict): The data to send with the request.

    Returns:
        Dict: The response from the API.
    """
    api_url = f"{base_api_url}{node_name}"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, json=payload)
        response.raise_for_status()
        api_response = response.json()
        return api_response

