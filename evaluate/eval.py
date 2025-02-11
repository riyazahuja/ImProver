import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from evaluate.metrics import *

"""
    Function to evaluate the correctness of a LLM Output
"""


def eval_correctness(thm: Theorem, sorries_are_errors=True):
    unit = thm.compile()
    msgs = unit.messages

    correct = (
        sum(1 for msg in msgs if "error" in msg)
        + sum(
            1
            for msg in msgs
            if "warning" in msg and "sorry" in msg and sorries_are_errors
        )
    ) == 0
    return (correct, msgs)


def eval_correctness_batched(thms: List[Theorem], sorries_are_errors=True):
    Theorem.compile_theorems(thms)
    output = [eval_correctness(thm, sorries_are_errors) for thm in thms]
    return output
