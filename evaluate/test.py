import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from evaluate.metrics import *

metric = length_metric()
example_selector = metric.get_example_selector()


example_prompt = PromptTemplate(
input_variables=["input", "output"],
template="<EXAMPLE>\nInput:\n{input}\n\nOutput:\n{output}\n</EXAMPLE>",
)

example_selector.k=1

examples_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="",
    input_variables=["item"],
)


#out = examples_prompt.format(item="Set")
out = examples_prompt.invoke({'item':'Set'}).text
for doc in out:
    print(doc.page_content)
    print('==========')
