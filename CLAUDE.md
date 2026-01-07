# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ImProver is an LLM-powered AI agent for automated proof optimization in Lean 4. The system orchestrates a pipeline that combines Lean metaprogramming, distributed LLM inference, and formal verification to automatically rewrite formal proofs according to user-defined metrics (length, readability, dependency minimization, etc.).

## Core Architecture

### Two-Language System
ImProver operates across **Lean 4** (for formal verification and metaprogramming) and **Python** (for ML inference and orchestration).

**Lean Side (TrainingData/, metrics/, lakefile.lean):**
- Compiles Lean source files and extracts proof structure via metaprogramming
- Evaluates proof correctness and metrics through the router pattern
- Provides goal state extraction and tactic analysis

**Python Side (ImProver/):**
- Orchestrates distributed inference using Ray/vLLM
- Manages prompt generation with RAG enhancement
- Handles evaluation, analysis, and data persistence (DuckDB/Parquet)

### Key Data Flow
```
Lean Code → Metaprogramming (TrainingData/) → Prompt Generation (ImProver/get_prompts/)
→ RAG Enhancement (optional) → Distributed Inference (ImProver/basic/inference.py)
→ Lean Evaluation (metrics/router.lean) → Analysis (ImProver/basic/analysis.py)
```

## Commands

### CLI Entry Point: `./improver`
All operations go through the unified CLI defined in `improver_cli.py`.

**Build Lean environment:**
```bash
lake build
```

**Generate prompts from dataset:**
```bash
./improver prompts get --dataset_path data/dataset.json --prompts_id "run_$(date +%s)" --split train
```

**Build RAG database (optional, adds 2+ hours):**
```bash
./improver rag build --dataset_path data/dataset.json --rag_id "rag_$(date +%s)" --model "embedding_model"
```

**Run inference pipeline:**
```bash
python ImProver/basic/improver.py \
  {metric_name} \
  data/dataset.json \
  {prompts_id} \
  --split train \
  --model "deepseek-ai/DeepSeek-Prover-V2-7B" \
  --gpus 8 \
  --n 5  # Best-of-5 sampling
```

**Evaluate generated proofs:**
```bash
python ImProver/basic/eval_improver.py {run_id}
```

**Analyze results:**
```bash
python ImProver/basic/analysis.py --run_id {run_id}
```

**Add a new metric:**
```bash
./improver metrics add \
  --name my_metric \
  --system_prompt "Instructions for optimization..." \
  --minmax min \
  --example_file metrics/my_metric/examples.lean
```

**Test a single file with Lean metaprogramming:**
```bash
lake exe get_prompts Module.Name output.json
lake exe eval_improver Module.Name metric_name run_id output.json
```

## Critical Architecture Details

### Lean Metaprogramming (TrainingData/)
- `Frontend.lean` - Compiles Lean files to extract `CompilationStep` structures (environment state before/after each command)
- `ExtractGoal.lean` - Extracts tactic-level goal states using the `better_extract_goal` tactic
- `TreeParser.lean` - Parses InfoTree structures to build proof dependency graphs
- These components enable ImProver to understand proof structure without modifying Lean core

### Metrics System (metrics/)
Each metric lives in `metrics/{metric_name}/`:
- `{metric_name}.lean` - Scoring function that takes CompilationStep and returns Float
- `config.json` - Full metric configuration (prompts, examples, scoring parameters)
- `examples.lean` - Tagged example theorems for few-shot learning
- `router.lean` - **Central dispatcher** that routes metric evaluation to the appropriate scorer

**Important:** When adding metrics, the router pattern ensures all scoring happens through `route_metric(name: String, cs: CompilationStep) -> Float` in `metrics/router.lean`.

### Prompt Generation Pipeline (ImProver/get_prompts/)
1. `get_prompts.py` orchestrates async processing of dataset files
2. For each file, spawns `lake exe get_prompts` (Lean executable)
3. Lean extracts theorems, generates prompts with context/annotations
4. Python calls `rag.py` for retrieval-augmented generation (if RAG enabled)
5. Results stored in `prompts/{prompts_id}/src/Repository/Module/File.json`

**Key insight:** Prompts are pre-generated and cached; inference reads from this cache.

### Distributed Inference (ImProver/basic/inference.py)
- Uses Ray for distributed execution with vLLM backend
- Automatic context window truncation when prompts exceed `max_model_len`
- Batch processing with configurable `max_num_batched_tokens` (default: 65536)
- Outputs to Parquet files + DuckDB table for efficient querying
- **Important parameters:**
  - `--gpus`: Number of GPUs (auto-detected)
  - `--num_blocks`: Parallelism level (typically 4x GPU count)
  - `--batch_size`: Inference batch size (reduce if OOM)
  - `--n`: Number of samples per theorem (best-of-N)

### Evaluation System (ImProver/basic/eval_improver.py)
- Spawns Lean subprocess per file: `lake exe eval_improver Module.Name metric run_id output.json`
- Lean compiles generated proof and routes to metric scorer
- Returns: `{og_score, new_score, delta, og_correct, new_correct, compile_error}`
- Async processing with semaphore-based concurrency control
- Results: `evals/{run_id}/evals/Repository/File.json` + `eval.duckdb`

### RAG System (ImProver/build/, ImProver/get_prompts/rag.py)
**Build Phase:**
1. `preprocess_rag.lean` - Extract declaration metadata (names, types, modules)
2. `informalize.py` - LLM converts formal proofs to informal statements/proofs
3. `build_db.py` - Embed informalized proofs using sentence-transformers, store in Chroma

**Retrieval Phase:**
- Query vector DB with theorem context + metadata filtering
- Return top-k similar informal proofs from compatible modules
- Inject results into prompts under `<RETRIEVED>...</RETRIEVED>` tags

## Dataset Format

Datasets are JSON files with train/test/validation splits:
```json
{
  "train": {
    "RepositoryName": [
      "path/to/File.lean",
      { "file": "path/with/specific.lean", "theorems": ["theorem_name1", "theorem_name2"] }
    ]
  },
  "test": { ... },
  "validation": { ... }
}
```

**Main dataset:** `data/final_dataset_fixed.json`
**Repositories integrated:** Mathlib, Improver_MIL, Compfiles, FLT, PFR, Foundation, Carleson, ConNF, Seymour, HepLean

## Built-in Metrics

- **length** (min): Minimize tactic count
- **readability** (max): LLM-evaluated on 6-point rubric (clarity, modularity, etc.)
- **declarativity** (max): Maximize "have" statements (forward reasoning)
- **dependency** (min): Minimize external theorem dependencies
- **completion** (max): Generate complete proofs without "sorry"
- **conjecturer**: Propose novel theorem variations

## Key Configuration Files

**Metric configs:** `metrics/{metric_name}/config.json`
- Contains: system_prompt, examples, scoring function, minmax, correctness conditions

**Inference configs:** YAML files in `configs/` or `experiments/`
- Example: `configs/main/len.yaml` for length optimization

**Lakefile:** `lakefile.lean`
- Defines Lean package dependencies and executables
- Required repositories: mil, compfiles, PFR, FLT, foundation, carleson, etc.

## File Locations

**Generated artifacts:**
- Prompts: `prompts/{prompts_id}/src/Repository/Module/File.json`
- Inference outputs: `evals/{run_id}/data/` (Parquet) + `data.duckdb`
- Evaluation results: `evals/{run_id}/evals/Repository/File.json` + `eval.duckdb`
- Analysis: `evals/{run_id}/analysis/`
- RAG databases: `rag/{rag_id}/informal_retrieval_db/` (Chroma)

**Code structure:**
- Lean metaprogramming: `TrainingData/`
- Python orchestration: `ImProver/`
- Metric definitions: `metrics/`
- Experiment scripts: `experiments/`

## Common Development Tasks

### Modifying Prompts
Edit `ImProver/get_prompts/get_prompts.py` for system prompt templates, context injection, or annotation formatting.

### Adding Custom Metrics
1. Create directory: `metrics/{metric_name}/`
2. Write Lean scorer: `metrics/{metric_name}/{metric_name}.lean`
   ```lean
   def my_metric_score (cs: CompilationStep): IO Float := ...
   ```
3. Create examples: `metrics/{metric_name}/examples.lean`
4. Run: `./improver metrics add --name {metric_name} --system_prompt "..." --minmax min`
5. Function auto-added to `metrics/router.lean`

### Debugging Evaluation Issues
- Check individual file: `lake exe eval_improver Module.Name metric run_id output.json`
- Review compilation errors in JSON outputs: `evals/{run_id}/evals/`
- Increase timeout in `eval_improver.py` (default: 20 min)

### Querying Results
```bash
# Using DuckDB CLI
duckdb evals/{run_id}/data.duckdb
> SELECT COUNT(*), AVG(new_score) FROM run_data WHERE new_correct = true;

# Using Python
import duckdb
con = duckdb.connect('evals/{run_id}/eval.duckdb')
df = con.execute('SELECT * FROM eval_data').df()
```

## Performance Tuning

**OOM errors:**
- Reduce `--batch_size` (default: 32)
- Increase `--num_blocks` (more parallelism, less memory per block)
- Reduce `--max_num_batched_tokens` (default: 65536)

**Slow inference:**
- Check GPU utilization: `nvidia-smi` should show >80%
- Increase `--num_blocks` if CPU-bound
- Ensure Ray cluster is configured correctly (check `127.0.0.1:8265`)

**Context window issues:**
- Automatic truncation activates when prompts exceed model's `max_model_len`
- Reserve space for generation tokens (default: 2048)
- Adjust in `inference.py` if needed

## Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI models and LLM-based metrics
- `NCCL_P2P_DISABLE=1` - Recommended for multi-GPU inference stability

## Important Constraints

- **Never amend git commits unless explicitly requested** - ImProver generates lots of experimental data
- **Lean compilation timeout:** 20 minutes per file (configurable in `eval_improver.py`)
- **Inference uses stop tokens:** `["</IMPROVED>"]` to terminate generation
- **Metrics with `minmax: "min"`:** Lower scores are better (length, dependency)
- **Metrics with `minmax: "max"`:** Higher scores are better (readability, declarativity)

## Testing Workflow

For quick testing with small datasets:
```bash
# Use tiny dataset (5-10 theorems, ~30 min total)
./improver prompts get --dataset_path data/final_dataset_fixed_tiny.json --prompts_id test --split train
python ImProver/basic/improver.py length data/final_dataset_fixed_tiny.json test --split train --model "deepseek-ai/DeepSeek-Prover-V2-7B" --n 1
python ImProver/basic/eval_improver.py {run_id}
python ImProver/basic/analysis.py --run_id {run_id}
```

## Expert Iteration

ImProver supports iterative self-improvement through expert iteration:
```bash
python experiments/expert_iteration.py \
  --inf-template configs/inference.yaml \
  --train-template configs/train.yaml \
  --base-model "deepseek-ai/DeepSeek-Prover-V2-7B" \
  --iterations 3
```

This automatically runs: inference → evaluation → training data generation → fine-tuning → next iteration.
