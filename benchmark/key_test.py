# from openai import OpenAI

# client = OpenAI()

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Write a haiku about recursion in programming."},
#     ],
# )

# print(completion.choices[0].message)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

output = llm.invoke("Hello how are you?")
print(output)
