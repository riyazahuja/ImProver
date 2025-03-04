import asyncio
import aiohttp
import json
import time
from vllm import LLM, SamplingParams
import sys

NUM_REQUESTS = 100  # Total number of concurrent requests
DUMMY_PROMPT = (
    "Shorten the current theorem (wrapped in <CURRENT>...</CURRENT>) to be as short as possible in length "
    "while also ensuring that the output is still a correct proof of the theorem. Include the output in the "
    "<IMPROVED>...</IMPROVED> tag.\n\n"
    "<CURRENT>\nDummy theorem content here\n</CURRENT>\n\n<IMPROVED>"
)
# Configuration
ENDPOINT = "http://0.0.0.0:8000/v1/chat/completions"  # Adjust port if needed


async def send_request(session, request_id):
    payload = {
        "model": "Llama-8B",
        "messages": [
            {"role": "user", "content": DUMMY_PROMPT}
        ],
        "max_tokens": 256
    }
    headers = {"Content-Type": "application/json"}
    try:
        async with session.post(ENDPOINT, json=payload, headers=headers) as response:
            resp_text = await response.text()
            # You might want to parse JSON here or log response info
            print(f"Request {request_id}: HTTP {response.status}")
            return resp_text
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None

async def main():
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(NUM_REQUESTS)]
        responses = await asyncio.gather(*tasks)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Completed {NUM_REQUESTS} requests in {elapsed:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
