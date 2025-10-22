# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ImProver is an LLM-powered AI agent for automated proof optimization in Lean 4. It allows arbitrary Lean code to be optimized for user-defined metrics (length, readability, etc.) by combining Lean metaprogramming, distributed GPU inference, and formal verification.

## Environment Setup

### Required Environment Variables
- `OPENAI_API_KEY`: Required for LLM-based metrics and certain RAG operations
- Optional GPU configuration through CLI flags (`--gpus`, `--tensor_parallel_size`)

### Lean Environment
- **Lean Version**: Specified in `lean-toolchain` file
- **Lake Configuration**: `lakefile.lean` defines dependencies on multiple Lean projects (mathlib, MIL, PFR, FLT, etc.)
- **Build Lean Environment**: `lake build` to compile Lean dependencies

### Python Environment
- No requirements.txt, setup.py, or pyproject.toml - dependencies must be installed manually
- Key dependencies: PyTorch, Ray, vLLM, DuckDB, Polars, ChromaDB
- Uses vLLM for distributed GPU inference with Ray

## Command-Line Interface

All commands go through `python improver_cli.py` with command groups:

### Common Commands

**Build RAG Database:**
```bash
python improver_cli.py rag build --dataset_path <path> --rag_id <id> --model <model_name>
```

**Generate Prompts:**
```bash
python improver_cli.py prompts get --dataset_path <path> --prompts_id <id> --rag_id <optional_rag_id>
```

**Run Inference:**
```bash
python improver_cli.py run inference --metric <metric_name> --dataset_path <path> --prompt_id <id> --model <model_name>
```

**Evaluate Results:**
```bash
python improver_cli.py run eval --run_id <run_id>
```

**Run Full Pipeline:**
```bash
python improver_cli.py run pipeline --metric <metric_name> --dataset_path <path> --prompt_id <id> --model <model_name>
```

**Create Custom Metric:**
```bash
python improver_cli.py metrics add --name <name> --system_prompt <prompt> --minmax <max|min>
```

### Knowledge Graph Commands
```bash
python improver_cli.py KG embed --prompts_id <id>
python improver_cli.py KG c3 --prompts_id <id> --kg_id <id>
python improver_cli.py KG make_db --dataset_path <path> --prompts_id <id> --kg_id <id>
python improver_cli.py KG filter --kg_id <id>
python improver_cli.py KG insert --kg_id <id> --neo4j_uri <uri>
```

### Configuration Files
Most commands accept `--config <path_to_yaml>` to load parameters from YAML files instead of CLI flags.

## Architecture

### Three-Phase Pipeline

**Phase 1: Data Preparation (CPU, 30min-2hr)**
- `TrainingData/` contains Lean metaprogramming code to extract theorem metadata
- `ImProver/get_prompts/` generates prompts with context, annotations, and RAG examples
- Output: JSON files in `evals/<prompts_id>/`

**Phase 2: Proof Generation (GPU, 1-24hr)**
- `ImProver/basic/inference.py` or `inference_server.py` runs distributed LLM inference
- Uses Ray + vLLM for GPU parallelization
- Output: Parquet files + DuckDB database in `evals/<run_id>/`

**Phase 3: Evaluation (CPU, 30min-4hr)**
- `ImProver/basic/eval_improver.py` verifies proofs with Lean compiler
- `ImProver/basic/analysis.py` computes metrics and delta improvements
- Output: JSON results in `evals/<run_id>/analysis/`

### Directory Structure

```
ImProver/
├── basic/           # Inference, evaluation, analysis pipeline
├── build/           # RAG preprocessing and database building
├── get_prompts/     # Prompt generation with context/RAG
├── metrics/         # Metric definitions and management
└── KG/              # Knowledge graph construction (deleted in current branch)

TrainingData/        # Lean metaprogramming for data extraction
├── Frontend.lean    # AST traversal and theorem extraction
├── TreeParser.lean  # Proof tree parsing
└── Utils/           # Helper functions

data/                # Training and test datasets (JSON)
prompts/             # Metric-specific prompt templates
metrics/             # Metric configurations (Lean + JSON)
experiments/         # Experiment scripts and configs
deepspeed_configs/   # DeepSpeed ZeRO optimization configs
```

### Key Components

**Metrics System (`ImProver/metrics/`)**
- Router pattern: `get_metric()` dispatches to appropriate metric module
- Each metric has: system prompt, examples, scoring function, comparison function
- Metrics can be code-based (length.lean) or LLM-based (with rubric)
- Add new metrics via `improver_cli.py metrics add` or by editing metric files

**RAG System (`ImProver/build/`)**
- `preprocess_rag.py`: Extracts theorem context and dependencies
- `informalize.py`: Generates natural language descriptions via LLM
- `build_db.py`: Creates ChromaDB vector store with embeddings
- RAG examples retrieved during prompt generation to aid optimization

**Prompt Generation (`ImProver/get_prompts/`)**
- Assembles prompts from: metric system prompt, theorem, context, annotations, RAG examples
- Context includes definitions/theorems used in proof
- Annotations interleave goal states as comments
- File context includes preceding definitions in source file

**Inference (`ImProver/basic/inference.py`)**
- vLLM engine with Ray for multi-GPU parallelization
- Automatic prompt truncation to fit model context window
- Batching and concurrent request handling
- Supports both local vLLM and Azure OpenAI (`inference_server.py`)

**Evaluation (`ImProver/basic/eval_improver.py`)**
- Asynchronous Lean compilation to verify correctness
- Metric computation via router
- Delta calculation: improvement = (original_score - optimized_score) for minimize metrics

**Analysis (`ImProver/basic/analysis.py`)**
- Aggregates results across dataset
- Computes success rates, average improvements
- Optionally generates training data (SFT, DPO, weighted SFT)

### Data Formats

**Dataset Format (JSON):**
```json
{
  "name": "theorem_name",
  "decl": "formal_lean_code",
  "context": ["dependency1", "dependency2"],
  "file_context": ["item1", "item2"],
  ...
}
```

**Prompt Format (stored in parquet):**
- `item_id`: Theorem identifier
- `prompt`: Full prompt text
- `metric`: Metric name
- `original_code`: Original Lean proof

**Inference Results (stored in parquet + DuckDB):**
- `item_id`, `output`, `valid`, `score`, `delta`, timestamps
- DuckDB allows SQL queries for analysis

## Development Workflows

### Testing a New Metric
1. Create metric: `python improver_cli.py metrics add --name <name> --system_prompt <prompt>`
2. Edit `metrics/<name>/metric.json` and `metrics/<name>/system.txt` as needed
3. Generate prompts: `python improver_cli.py prompts get --dataset_path <path>`
4. Run inference: `python improver_cli.py run inference --metric <name> --dataset_path <path> --prompt_id <id>`
5. Evaluate: `python improver_cli.py run eval --run_id <run_id>`
6. Analyze: `python improver_cli.py run analysis --run_id <run_id>`

### Running Experiments
1. Create experiment script in `experiments/` (see existing `.sh` files)
2. Define config YAML with all parameters
3. Use `--config` flag to load parameters
4. Results stored in `evals/<run_id>/`

### Working with Lean Code
- Lean libraries defined in `lakefile.lean`
- Executable targets: `get_prompts`, `get_examples`, `eval_improver`, `preprocess_rag`
- Build: `lake build <target>`
- Run: `lake exe <target> <args>` or `.lake/build/bin/<target> <args>`

### Modifying Metaprogramming
- Edit files in `TrainingData/`
- Rebuild: `lake build TrainingData`
- Main entry points: `Frontend.lean` (extraction), `TreeParser.lean` (proof trees)

### GPU Resource Management
- Default: Auto-detects GPU count with `torch.cuda.device_count()`
- Override: `--gpus N` flag
- Tensor parallelism: `--tensor_parallel_size N` for multi-GPU models
- Ray configuration: `--concurrency`, `--engine_gpu_resources`, `--engine_cpu_resources`

## Important Notes

### Lean Compilation
- The evaluation phase spawns Lean processes to verify proofs
- Each proof compiled independently in temporary environment
- Failures captured as `valid=False` in results
- Timeout controlled by subprocess timeout (not currently configurable via CLI)

### Memory Management
- vLLM uses PagedAttention for efficient KV cache
- Prompt truncation: `--truncate_prompt_tokens` (default 14336)
- Max output: `--max_tokens` (default 2048)
- Batch size: `--batch_size` (default 32)

### Parallel Processing
- CPU parallelism: `--cpus` (defaults to all cores)
- GPU parallelism: Handled by Ray + vLLM
- Prompt generation is embarrassingly parallel across CPUs
- Inference batches requests across GPUs

### File ID Conventions
- `prompts_id`: Identifier for prompt generation run (defaults to timestamp)
- `rag_id`: Identifier for RAG database
- `run_id`: Identifier for inference run (defaults to timestamp)
- `kg_id`: Identifier for knowledge graph
- Results stored in `evals/<id>/` directories

### Deleted Components (Current Branch)
The git status shows many deleted files in `ImProver/KG/`, `ImProver/cotraining/`, `train/`, and `docs/`. These are still referenced in the CLI but may not work. Check `dev` branch history or `main` branch if these are needed.

## Common Pitfalls

1. **Missing Environment Variables**: Set `OPENAI_API_KEY` for LLM metrics
2. **GPU OOM**: Reduce `--batch_size`, `--max_model_len`, or `--tensor_parallel_size`
3. **Lean Compilation Errors**: Ensure `lake build` succeeded and dependencies are available
4. **RAG Database Not Found**: Must run `rag build` before using `--rag K` in prompts
5. **Metric Not Found**: Check `metrics/<name>/metric.json` exists and is valid JSON
6. **YAML Config Overrides**: CLI flags are ignored when `--config` is used; all params must be in YAML
