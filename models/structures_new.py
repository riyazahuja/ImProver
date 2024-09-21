from __future__ import annotations
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from typing import List, Union, Optional, Tuple
import json
import tempfile
import subprocess
from textwrap import indent
import re
import sys
import pathlib

sys.path.append(str(Path(__file__).parent.parent))
from models.rag import *

# NOTE position encodings have top left as line 1, column 0
# -> we restandardize to line 1 column 1. makes more sense that way


# Improve by using (annotated) theorems and files.
class Dependency(BaseModel):
    dependency: str = Field(
        description="Constant term in theorem dependent on external module"
    )
    src_file: str = Field(description="Source file of dependency")
    src_content: str = Field(
        description="source content (decl, def, proof, etc), just a few lines"
    )
    explicit: bool = Field(
        description="Is dependency explicit in the pp form of the parent theorem"
    )
    direct: bool = Field(description="is dependency direct or unfolded")
    kind: str = Field(
        description="What type of object is the dependency (theorem, def,etc)"
    )


class AnnotatedProofStep(BaseModel):
    prevState: List[str] = Field(
        description="Pretty printed tactic state before the tactic invocation"
    )
    tactic: str = Field(description="One line/tactic in a tactic proof.")
    nextState: List[str] = Field(
        description="Pretty printed tactic state after the tactic invocation"
    )
    # srcUpToTactic: str = Field(
    #     description="Source code from file start to current tactic"
    # )
    # declUpToTactic: str = Field(
    #     description="Source code from theorem declaration to current tactic"
    # )
    start: Tuple[Optional[int], Optional[int]] = Field(
        description="start coordinates from source file as (row,column)"
    )
    end: Tuple[Optional[int], Optional[int]] = Field(
        description="end coordinates from source file as (row,column)"
    )


class Message(BaseModel):
    severity: str = Field(description="Message severity")
    start: Tuple[Optional[int], Optional[int]] = Field(
        description="start coordinates from source file as (row,column)"
    )
    end: Tuple[Optional[int], Optional[int]] = Field(
        description="end coordinates from source file as (row,column)"
    )
    message_src: Optional[str] = Field(
        description="equivalent to source_contents[start:end]"
    )
    content: str = Field(description="Message contents")


class AnnotatedTheorem(BaseModel):
    decl: str = Field(description="Theorem declaration")
    # declID: str = Field(description="Unique theorem declaration ID")
    # src: str = Field(description="Source repo of the theorem")
    # leanFile: str = Field(description="Lean file in which theorem is located")
    context: str = Field(
        description="Context of the theorem (i.e. file contents up to decl)"
    )
    proof: List[AnnotatedProofStep] = Field(
        ..., description="Sequence of annotated proofsteps for full proof of theorem."
    )
    # project_path: str = Field(description="Local path to src repo contents")
    messages: List[Message] = Field(..., description="Messages from the lean server")
    pretty_print: str = Field(description="Content of theorem src file.")
    # proof_tree: List[Tuple[str, List[int], List[int]]] = Field(
    #     description="data for efficient proof tree construction"
    # )
    dependencies: List[Dependency] = Field(
        description="Theorem dependencies from all imported modules"
    )


class Metric(BaseModel):
    name: str
    prompts: List[Tuple[str, str]]
    examples: List[str]
    minmax: str

    def get_example_selector(self):
        if len(self.examples) == 0:
            return None

        vs = get_metric_vs(self.examples, self.name)
        return vs


class PromptResult(BaseModel):
    content: str
    correct: Optional[bool]
    score: Optional[Union[float, int]]
    delta: Optional[Union[float, int]]


class PromptRequest(BaseModel):
    previous_data: List[PromptResult]
    theorem: AnnotatedTheorem
    metric: Metric
    output_format: str
    model: str
    annotation: bool
    rag: int
    examples: int
    improved_context: bool
    token: bool


def parse_prompt_request(data):
    theorem_raw = data["theorem"]
    prev_raw = data["previous_data"]
    metric_raw = data["metric"]
    output_key = data["output_key"]

    theorem = AnnotatedTheorem.__pydantic_model__.parse_raw(theorem_raw)
    metric = Metric.__pydantic_model__.parse_raw(metric_raw)
    prev_data = [PromptResult.__pydantic_model__.parse_raw(d) for d in prev_raw]

    return PromptRequest(
        theorem=theorem,
        metric=metric,
        previous_data=prev_data,
        output_format=output_key,
    )
