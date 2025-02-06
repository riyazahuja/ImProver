import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))
from models.structures import *
from evaluate.metrics import *

"""
    Function to evaluate the correctness of a LLM Output
"""


def eval_correctness(thm, sorries_are_errors=True):
    if type(thm) == AnnotatedTheorem:
        new_thm = thm
    elif type(thm) == Theorem:
        new_thm = annotateTheorem(thm)
    else:
        raise ValueError(
            f"Input is not a Theorem/AnnotatedTheorem obj:\nthm:\n{thm[:5]}\ntype: {type(thm)}"
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


def eval_correctness_batched(thms, sorries_are_errors=True):

    if any([type(thm) != Theorem and type(thm) != AnnotatedTheorem for thm in thms]):
        raise ValueError(
            f"Input is not a Theorem/AnnotatedTheorem obj:\nthm:\n{thm}\ntype: {type(thm)}"
        )

    if all([type(thm) == AnnotatedTheorem for thm in thms]):
        thms_new = thms
    else:
        thms_enum = [(i, thm) for i, thm in enumerate(thms)]
        unannotated = [item for item in thms_enum if type(item[1]) == Theorem]
        annotated = {
            item[0]: item[1] for item in thms_enum if type(item[1]) == AnnotatedTheorem
        }
        annotated_raw = annotateTheorems([thm for _, thm in unannotated])
        for i in range(len(annotated_raw)):
            annotated[unannotated[i][0]] = annotated_raw[i]

        thms_new = [annotated[i] for i in range(len(thms))]

    output = []
    for new_thm in thms_new:

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
        output.append((correct, msgs, new_thm))
    return output
