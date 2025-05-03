import asyncio
import aiohttp
import json
import time
from vllm import LLM, SamplingParams
import sys
import json
import traceback

async def send_request(session, request_id, data, endpoint):
    headers = {"Content-Type": "application/json"}
    try:
        async with session.post(
            endpoint, json=json.loads(data), headers=headers
        ) as response:
            resp_text = await response.text()

            # print(f"Request {request_id}: HTTP {response.status}")
            return (True, resp_text)
    except Exception as e:
        # print(f"Request {request_id} failed: {e}")
        return (False, f"Request {request_id} failed: {e}\nTB: {traceback.format_exc()}")


async def main(data, n, endpoint):
    n = int(n)
    start_time = time.time()
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [send_request(session, i, data, endpoint) for i in range(n)]
        responses = await asyncio.gather(*tasks)
    end_time = time.time()
    elapsed = end_time - start_time
    output = []
    for rec, response_json in responses:
        try:
            if rec:
                print(response_json)
                response = json.loads(response_json)
                content = response["choices"][0]["message"]["content"]
                output.append(content)
            else:
                print(f">>> Error in model output:\n")
                print(f">>> \tError: {response_json}\n")
        except:
                print(f">>> Error in model output [weird case]:\n")
                print(f">>> \tError: {response_json}\n")
                continue    
    print("<RESPONSE>")
    print(json.dumps(output))
    # print(f"Completed {n} requests in {elapsed:.2f} seconds")
    # print("RESPONSES:")
    # print(f"{type(responses)}")
    # print(f"{responses[:100]}")


if __name__ == "__main__":
    print("Python has been entered!!")
    prompt = "What is Sard's theorem? Think about it first before giving a consise answer."
    # if len(args) != 3:
    #     print("usage: python3 stress_test.py <data> <n> <endpoint>")
    data = {"model": "Qwen-7B-ALL",
      "messages": [{"role": "user", "content": prompt}],
      "max_tokens": 512
    }
    asyncio.run(main(json.dumps(data), 1, "http://0.0.0.0:8003/v1/chat/completions"))
