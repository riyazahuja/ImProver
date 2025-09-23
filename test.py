import os
import requests
import json


def call_azure_openai_model():
    # Get API key from environment variable
    # api_key = os.getenv("AZURE_API_KEY")
    # if not api_key:
    #     raise ValueError("AZURE_API_KEY environment variable not set")

    # API endpoint
    url = "https://riyaz-mfrbnakc-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

    # Headers
    headers = {"Authorization": f"Bearer {api_key}"}

    # Request payload
    data = {
        "messages": [
            {"role": "user", "content": "I am going to Paris, what should I see?"}
        ],
        "max_tokens": 1024,
        # "temperature": 1,
        # "top_p": 1,
        "model": "DeepSeek-R1-0528",
    }

    # Make the request
    response = requests.post(url, headers=headers, json=data)
    # print(response.__dict__)
    # Check if request was successful
    response.raise_for_status()

    # Return the response JSON
    return response.json()


# Example usage
if __name__ == "__main__":
    try:
        result = call_azure_openai_model()
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
