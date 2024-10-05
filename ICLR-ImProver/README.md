# ImProver: Agent-Based Automated Proof Optimization

## Overview
ImProver is a LLM-powered AI agent for proof rewriting tasks, built around a general-purpose neural theorem proving framework. It allows for arbitrary Lean 4 code to be optimized for an arbitrary metric, enabling users to automatically optimize their formal proofs by providing a lambda and prompt. ImProver leverages advanced techniques in Lean metaprogramming, machine learning, and formal methods to deliver optimized proofs that meet user-defined criteria.

ImProver automates the process of optimizing Lean 4 proofs through the following steps:
1. **Configuration and Setup**: Define configuration files specifying Lean projects.
2. **Building Process**: Use Lean metaprogramming to generate proof data and cache it in JSON format.
3. **Prompt Generation**: Construct prompts using templates refined with theorem and metric data.
4. **Retrieval-Augmented Generation (RAG)**: Enhance prompts with relevant examples and documentation.
5. **Proof Generation and Refinement**: Generate and iteratively refine proofs using language models.
6. **Evaluation**: Assess the correctness and quality of proofs using predefined and user-defined metrics.
7. **Benchmarking**: Compare different optimization strategies to determine the most effective approach.

## Repository Structure
```
root/
├── .cache/ # Build Cache
├── .db/ # Vector DB Cache
├── .lake/ # Lean Environment
├── .trees/ # Proof tree export images
├── benchmark/ # Benchmarking tools
│ └── data/ # Benchmarking output
├── configs/ # Build configurations
├── evaluate/ # Correctness and metric evaluation
├── models/ # LLM interface
├── scripts/ # Build scripts
├── TrainingData/ # Metaprogramming
├── lake-manifest.json # Dependency info
├── lakefile.lean # Lean config
├── lean-toolchain # Lean version
└── README.md
```

## Key Files
- **scripts/build.py**: Build environment and cache
- **benchmark/tools.py**: Run benchmarks with file-specified configuration.
- **benchmark/extract.py**: Run experimental measurements over benchmarking data.
- **models/prompt.py**: All prompting chains
- **models/rag.py**: All RAG code
- **models/structures.py**: All objects and datastructures
- **evaluate/metrics**: Metric definitions.
- **evaluate/build_prooftree**: Proof tree generation

## Configuration
Configuration files specify the details of the Lean projects to be built. They are JSON arrays of objects, each representing a Lean project. Here is an example configuration file:

```json
[
    {
        "path": "/Users/user/Desktop/lean-project",
        "lean": "leanprover/lean4:v4.9.0",
        "name": "LeanProject",
        "import_file": "LeanProject.lean",
        "imports": ["LeanProject"]
    },
    {
        "repo": "https://github.com/leanprover-community/mathlib4",
        "commit": "v4.9.0",
        "lean": "leanprover/lean4:v4.9.0",
        "name": "mathlib",
        "import_file": "Mathlib.lean",
        "imports": ["Mathlib"],
        "build": false
    }
]
```

## Metrics
ImProof comes with several preinstalled metrics for evaluating proofs:

- **Length Metric**: Measures the length of the proof in terms of the number of steps.
- **Readability Metric**: Evaluates the degree of modularity by analyzing the proof tree for reusable, independent subproofs.
- **Similarity Metric**: Measures the structural and syntactic differences between two proofs. (Not used in ICLR)
- **Completion Metric**: Measures the completeness of a proof by evaluating the number of errors present. This is essentially used for full proof generation
### Adding Custom Metrics
To add a new metric, create an instance of the `Metric` class by defining the following parameters:

```python
Metric(name="Metric Name",
      prompts=prompts,
      examples=examples,
      minmax="minimize or maximize",
      score_fn=score_fn, #default = None; unary
      metric_fn=metric_fn, #default = None; binary
      cmp=comparison_fn, #default = None
      lock_refinement_state=False #default = False; Refinement compares to orginal or most recent instance
      )
```

## Usage
- **Setup the Environment**: Clone the repository and set up the Lean environment by defining configuration files. Set the `OPENAI_API_KEY` environment variable.
- **Build and Cache Proof Data**: Run the build scripts to generate and cache proof data in JSON format.
   - `python scripts/build.py --config CONFIG_PATH`
- **Setup Run Configuration**: Use the provided tools and models to generate the parameter tuning for your desired tests.
- **Evaluate and Benchmark**: Assess the correctness and quality of the generated proofs using the evaluation and benchmarking tools.
   - `python benchmark/tools.py`

## Acknowledgements
We would like to thank Kim Morrison for the [Training Data repository](https://github.com/semorrison/lean-training-data) and Sean Welleck for the [Neural Theorem Proving (NTP) toolkit repository](https://github.com/cmu-l3/ntp-toolkit), which served as foundational resources for this project. Additionally, we would like to thank the Paperproof team for the [Paperproof repository](https://github.com/Paper-Proof/paperproof), which paved the way for our own prooftree generation and analysis system.

## Contributing
We welcome contributions from the community. If you have suggestions, bug reports, or want to contribute code, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
