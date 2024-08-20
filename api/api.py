import httpx
from typing import Dict

async def call_api(node_name: str, payload: Dict) -> Dict:
    """
    Function to call an API with a given URL, payload, and node name.

    Args:
        node_name (str): The name of the node to include in the URL.
        payload (Dict): The data to send with the request.

    Returns:
        Dict: The response from the API.
    """
    # Format the URL to include the node_name
    api_url = f"http://localhost:8000/api/v1/private/applications/endpoint/{node_name}"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, json=payload)
        response.raise_for_status()
        api_response = response.json()
        return api_response

