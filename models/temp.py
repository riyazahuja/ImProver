

from __future__ import annotations
from langchain.globals import set_verbose,set_debug
set_debug(True)
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import sys
#print(sys.path)
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from models.rag import *
from evaluate.metrics import *
from evaluate.eval import eval_correctness
from concurrent.futures import ThreadPoolExecutor
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
import logging
from typing import Final
import tiktoken




metric = length_metric()
examples = len(metric.examples)



ex = [
    {
        "input": example['input'].replace(r'{',r'{{').replace(r'}',r'}}'),
        "output": example['output'].replace(r'{',r'{{').replace(r'}',r'}}')
    }
    for example in metric.examples
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    ex,#metric.examples,
    OpenAIEmbeddings(),
    Chroma,
    k=min(examples,len(metric.examples)),
)


def get(content,i):
    example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="<EXAMPLE>\nInput:\n{input}\n\nOutput:\n{output}\n</EXAMPLE>",
    )

    example_selector.k=i

    examples_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix="",
        input_variables=["item"],
    )


    #out = examples_prompt.format(item="Set")
    out = examples_prompt.invoke({'item':content}).text
    print(f'{i}: \n{out}\n===============')


with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(get,'Set',1) for _ in range(5)]

