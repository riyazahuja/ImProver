import asyncio
import aiohttp
import json
import time
from vllm import LLM, SamplingParams
import sys
import json



async def send_request(session, request_id, data, endpoint):
    headers = {"Content-Type": "application/json"}
    try:
        async with session.post(endpoint, json=json.loads(data), headers=headers) as response:
            resp_text = await response.text()

            # print(f"Request {request_id}: HTTP {response.status}")
            return resp_text
    except Exception as e:
        # print(f"Request {request_id} failed: {e}")
        return None
    
    

async def main(data, n, endpoint):
    n=int(n)
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i, data, endpoint) for i in range(n)]
        responses = await asyncio.gather(*tasks)
    end_time = time.time()
    elapsed = end_time - start_time
    output = []
    for response_json in responses:
        try:
            response = json.loads(response_json)
            content = response["choices"][0]['message']['content']
            output.append(content)
        except:
            continue
    print('<RESPONSE>')
    print(json.dumps(output))
    # print(f"Completed {n} requests in {elapsed:.2f} seconds")
    # print("RESPONSES:")
    # print(f"{type(responses)}")
    # print(f"{responses[:100]}")

if __name__ == "__main__":
    print("Python has been entered!!")
    args = sys.argv[1:]
    if len(args) != 3:
        print('usage: python3 stress_test.py <data> <n> <endpoint>')
    
    asyncio.run(main(*args))
