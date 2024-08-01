import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from evaluate.metrics import *


def eval_correctness(thm, sorries_are_errors=True):
    if type(thm) == AnnotatedTheorem:
        new_thm = thm
    elif type(thm) == Theorem:
        new_thm = annotateTheorem(thm, force=True)
    else:
        raise ValueError(
            f"Input is not a Theorem/AnnotatedTheorem obj:\nthm:\n{thm}\ntype: {type(thm)}"
        )
    msgs = new_thm.messages

    correct = (
        sum(1 for msg in msgs if msg.severity == "error")
        + sum(
            1
            for msg in msgs
            if msg.severity == "warning"
            and "sorry" in msg.content
            and sorries_are_errors
        )
    ) == 0
    return (correct, msgs, new_thm)
