# ImProver: System Architecture Overview

## High-Level Summary

ImProver is an LLM-powered AI agent for automated proof optimization in Lean 4. The system orchestrates a sophisticated pipeline that combines Lean metaprogramming, machine learning inference, and formal verification to automatically rewrite proofs according to user-defined metrics (length, readability, dependency minimization, etc.).

## Core Architectural Components

### 1. **Lean 4 Integration (TrainingData/)**
The foundation of ImProver's proof manipulation capabilities.

**Key Files:**
- `TrainingData/Frontend.lean` - Lean source compilation infrastructure
  - Entry point: `processInput()` - Compiles Lean source code to Environment + Messages + InfoTrees
  - Returns `CompilationStep` structures with environment state before/after each command
  - Handles partial compilation via optional existing Environment

- `TrainingData/ExtractGoal.lean` - Tactic state extraction and formatting
  - `stateAsSignature()` - Extracts goal states in signature-like format
  - `better_extract_goal` tactic - Produces annotated proof states

- `TrainingData/TreeParser.lean` - Proof structure parsing
  - Parses InfoTree structures to extract proof dependencies and structure

- `TrainingData/Utils/` - Utility modules for theorem metadata, imports, etc.

**Purpose:** Enables Lean metaprogramming to:
- Extract proof structure and metadata
- Analyze dependencies and imports
- Generate proof trees
- Access goal states at each tactic step

### 2. **Data Structures & Datasets (data/, ImProver/)**

**Dataset Format:**
```json
{
  "train": {
    "Repository_Name": [
      "path/to/File.lean",
      { "file": "path/with/theorems.lean", "theorems": ["specific_theorem_names"] }
    ]
  },
  "test": { ... },
  "validation": { ... }
}
```

**Key Dataset Files:**
- `final_dataset_fixed.json` - Main dataset with file paths organized by repository
- `train_test_split.json` - Train/test split definitions
- Example datasets: FLT, Foundation, Compfiles, PFR, Carleson, HepLean (Physics Lean)

**Repositories Integrated:**
- Lean Mathlib
- Improver_MIL (custom)
- Compfiles
- FLT (Fermat's Last Theorem)
- PFR (Probabilistic Framework)
- Foundation (Logical foundations)
- HepLean (Physics)

### 3. **Metrics System (ImProver/metrics/, metrics/)**

**Architecture:**
- Each metric is a directory: `metrics/{metric_name}/`
- Components:
  - `config.json` - Metric configuration (prompts, scoring functions, examples)
  - `examples.lean` - Lean file with tagged example theorems
  - `examples.json` - Extracted example data
  - `{metric_name}.lean` - Lean scoring function

**Metric Configuration Structure:**
```json
{
  "name": "metric_name",
  "scoring": {
    "score_fn": "metrics/metric/metric.lean",
    "sorry_ok": false,
    "correctness_condition": "eq|not_equal|compiles|none",
    "minmax": "min|max",
    "input_sorry": false
  },
  "examples": { ... },
  "prompts": { ... },
  "llm": { "llm_metric": false, "metric_model": null, "rubric": null }
}
```

**Router Pattern:**
- `metrics/router.lean` - Central router that dispatches metric evaluation
- Example: `route_metric(name: String, cs: CompilationStep) -> Float`
- Supports custom metrics via the addition pipeline

**Built-in Metrics:**
1. **Length** - Minimizes tactic count
2. **Readability** - Maximizes modularity + clarity (LLM-evaluated with rubric)
3. **Declarativity** - Maximizes explicit "have" statements
4. **Dependency** - Minimizes external theorem dependencies
5. **Completion** - Full proof generation (sorry-free)
6. **Conjecturer** - Proposes novel theorem variations

### 4. **Prompt Generation Pipeline (ImProver/get_prompts/)**

**Workflow:**

```
Dataset → Lean Metaprogramming → Prompt Extraction → JSON Storage
   ↓             ↓                    ↓
(files.json)  get_prompts.lean   get_prompts.py   (prompts/)
```

**Key Files:**
- `get_prompts.lean` - Lean executable that:
  - Compiles each file in the dataset
  - Extracts human theorems
  - Generates prompts per theorem
  - Calls Python for RAG enrichment
  
- `get_prompts.py` - Main Python orchestrator:
  - Async processing of files using asyncio
  - Executes `lake exe get_prompts` for each file
  - Calls `rag.py` for retrieval-augmented generation
  - Stores results in `prompts/{prompts_id}/src/`

- `utils.lean` - Utilities for theorem extraction
- `index.lean` - Index-building for retrieval
- `where_with_end.lean` - Structured formatting

**Output Structure:**
```
prompts/{prompts_id}/
├── config.json
└── src/
    └── Repository/Module/File.json
```

### 5. **RAG System (Retrieval-Augmented Generation)**

**Two-Phase RAG Pipeline:**

**Phase 1: RAG Building (ImProver/build/)**
1. `preprocess_rag.lean` - Extracts declaration metadata from Lean files
   - Declaration names, types, modules
   - Creates `decl_data.json` per repository
   
2. `informalize.py` - Informalizes formal proofs using LLM
   - Takes formal Lean proofs → generates informal statements + proofs
   - Uses batch inference with Ray
   - Outputs `informal_data` to DuckDB database
   
3. `build_db.py` - Constructs vector database
   - Embeds informal statements + proofs using HuggingFace embeddings
   - Stores in Chroma vector DB (`informal_retrieval_db`)
   - Stores module metadata in DuckDB

**Phase 2: Retrieval (ImProver/get_prompts/rag.py)**
- Receives query: theorem name, module, proof text
- Queries Chroma vector store with metadata filtering
- Returns top-k most similar informal proofs from compatible modules
- Results injected into prompts

**Storage Structure:**
```
rag/{rag_id}/
├── informal_data.duckdb      # Informal statements/proofs
├── data.duckdb               # Module metadata
├── informal_retrieval_db/    # Chroma vector index
├── decl_data.json            # Declaration metadata per repo
└── module_data.json          # Module metadata per repo
```

### 6. **Inference & Generation (ImProver/basic/)**

**Inference Pipeline:**

```
Prompts → Model Loading → Batch Generation → Output Storage
            ↓
       Ray/vLLM Distributed Inference
```

**Key Files:**
- `inference.py` - Main inference orchestrator
  - Uses Ray with vLLM for distributed inference
  - Handles tensor parallelism across GPUs
  - Manages context window truncation
  - Implements sampling parameters (temperature, top_p, repetition_penalty)
  - Output: Parquet files + DuckDB database

- `inference_server.py` - HTTP server for inference
- `inference_http.py` - HTTP client for inference

**Inference Configuration:**
```python
# Sampling parameters
temperature = 0.3
top_p = 0.9
repetition_penalty = 1.05
max_tokens = 2048
stop_tokens = ["</IMPROVED>"]

# Ray vLLM Configuration
tensor_parallel_size = 1
enable_chunked_prefill = True
max_model_len = 16384
max_num_batched_tokens = 65536
```

**Data Flow:**
```
Raw Prompts (DataFrame)
  ↓
Truncation + Batching (Ray)
  ↓
vLLM Processing
  ↓
Parquet Output (evals/{run_id}/data/)
  ↓
DuckDB Table (evals/{run_id}/data.duckdb)
```

### 7. **Evaluation System (ImProver/basic/)**

**Evaluation Workflow:**

```
Generated Proofs → Lean Evaluation → JSON Results → Analysis
                        ↓
                  eval_improver.lean
                  (Lean executable)
```

**Key Files:**
- `eval_improver.py` - Async evaluation orchestrator
  - Spawns Lean subprocess for each file
  - Compiles generated proofs
  - Applies metric scoring functions
  - Produces JSON evaluation results

- `eval_improver.lean` - Lean executable that:
  - Loads generated proofs
  - Compiles them for correctness checking
  - Routes to appropriate metric scorer
  - Returns success/failure + score

**Evaluation Process:**
1. Load generated proof from inference results
2. Attempt to compile in Lean
3. Extract compilation results (success/failure)
4. Apply metric scoring function
5. Calculate delta (improvement) vs. original
6. Store results: `evals/{run_id}/evals/{File}.json`

**Output Structure:**
```
evals/{run_id}/
├── config.json              # Configuration used
├── data/                    # Inference outputs (Parquet)
├── data.duckdb             # DuckDB table of results
├── evals/                  # Evaluation results per file
│   └── Repository/File.json
└── eval.duckdb             # Aggregated evaluation table
```

### 8. **Analysis & Metrics Reporting (ImProver/basic/analysis.py)**

**Metrics Tracked:**
- **Accuracy**: Fraction of successfully compiled proofs
- **Nonzero Accuracy**: Accuracy where delta is in desired direction
- **Improvement**: Average delta (improvement magnitude)
- **Improvement Rate**: Success rate for improvement attempts
- **Best-of-N Performance**: Aggregates results across multiple samples

**Output Analysis Structure:**
```
evals/{run_id}/analysis/
├── BoN/
│   ├── raw_data.json
│   ├── training_data.json
│   ├── final_stats.json
│   └── success_map.json
└── detailed_metrics/
    ├── by_file/
    ├── by_theorem/
    └── summary.json
```

## Data Flow: Complete End-to-End Pipeline

### Phase 1: Data Preparation
```
Repository Code (Lean)
    ↓
Dataset Definition (JSON)
    ↓
Lean Metaprogramming Extraction
    ├─→ Theorem metadata
    ├─→ Dependencies
    └─→ Goal states
    ↓
Prompt Generation
    ├─→ System prompts (metric-specific)
    ├─→ Theorem context
    └─→ Annotations
    ↓
RAG Enhancement
    ├─→ Informal proofs (via LLM)
    ├─→ Vector embeddings
    ├─→ Retrieval (top-k similar)
    └─→ Injection into prompts
    ↓
Prompt Storage (JSON)
```

### Phase 2: Proof Generation
```
Dataset + Prompts + RAG Cache
    ↓
Ray/vLLM Batch Inference
    ├─→ Model loading
    ├─→ Distributed inference
    ├─→ Sampling (temperature, top_p)
    └─→ Output generation
    ↓
Generated Proofs (Parquet + DuckDB)
```

### Phase 3: Evaluation & Metrics
```
Generated Proofs
    ↓
Lean Compilation Check
    ├─→ Correctness verification
    └─→ Parsing proof structure
    ↓
Metric Scoring
    ├─→ Route to metric function
    ├─→ Calculate score
    └─→ Compare to original (delta)
    ↓
Evaluation Results (JSON)
    ↓
Analysis & Aggregation
    ├─→ Per-file statistics
    ├─→ Per-metric summaries
    ├─→ Overall improvements
    └─→ Success maps
```

## Key Design Patterns

### 1. **Async/Concurrent Processing**
- Uses `asyncio` for CPU-bound Lean operations
- Ray for distributed GPU inference
- Semaphore-based concurrency control

### 2. **Configuration-Driven**
- Metrics defined via JSON config + Lean code
- Prompts parameterized with templates
- Easy addition of new metrics via `metrics add` CLI

### 3. **Lazy Evaluation with MLList**
- Lean compilation uses lazy lists
- Incremental processing of large codebases
- Memory-efficient for large projects

### 4. **Database Abstraction**
- DuckDB for structured data access
- Parquet for distributed storage
- Schema-first with JSON metadata

### 5. **Modular Scoring**
- Lean metaprogramming for correctness
- Python LLM metrics for subjective evaluation
- Pluggable scoring functions via router pattern

## Important File Formats

### Dataset Format
```json
{
  "train|test|validation": {
    "RepoName": [
      "File.lean" | { "file": "File.lean", "theorems": ["names"] }
    ]
  }
}
```

### Prompt JSON Format
```json
{
  "theorem_name": {
    "original": { "statement": "...", "proof": "..." },
    "context": { "imports": [...], "definitions": [...] },
    "raw_prompt": "system + context + examples + current proof",
    "system_prompt": "metric-specific instructions",
    "annotation": "goal state comments",
    "rag": [{ "content": "...", "metadata": {...} }],
    "examples": [{ "before": "...", "after": "..." }]
  }
}
```

### Evaluation Result Format
```json
{
  "file": "Path.lean",
  "theorem": "theorem_name",
  "og_proof": "original proof",
  "generated_proof": "generated proof",
  "og_score": 10.0,
  "new_score": 8.0,
  "delta": -2.0,
  "og_correct": true,
  "new_correct": true,
  "compile_error": null
}
```

## Common Development Workflows

### 1. Adding a New Metric
```bash
# Define metric
./improver metrics add \
  --name "my_metric" \
  --system_prompt "Custom instructions" \
  --minmax "min|max" \
  --example_file "metrics/my_metric/examples.lean"

# Implement scoring function
# metrics/my_metric/my_metric.lean
def my_metric_score(cs: CompilationStep): IO Float := ...

# Update router
# metrics/router.lean - add route case
```

### 2. Running a Full Optimization Pipeline
```bash
# 1. Generate prompts
./improver prompts get \
  --dataset_path data/dataset.json \
  --prompts_id "my_run_$(date +%s)" \
  --split "train"

# 2. Build RAG (optional)
./improver rag build \
  --dataset_path data/dataset.json \
  --rag_id "my_rag" \
  --model "embedding_model"

# 3. Run inference
python ImProver/basic/improver.py \
  "length" data/dataset.json prompts_id \
  --model "deepseek-ai/DeepSeek-Prover-V2-7B" \
  --n 5  # best-of-5

# 4. Evaluate results
python ImProver/basic/eval_improver.py run_id

# 5. Analyze
python ImProver/basic/analysis.py --run_id run_id
```

### 3. Expert Iteration Loop
```bash
# Automated loop: inference → evaluation → training
python experiments/expert_iteration.py \
  --inf-template configs/inference.yaml \
  --train-template configs/train.yaml \
  --base-model "deepseek-ai/DeepSeek-Prover-V2-7B" \
  --iterations 3
```

## System Dependencies

**Core:**
- Python 3.8+
- Lean 4 (via lake)
- PyTorch/Transformers

**Inference:**
- Ray (distributed computing)
- vLLM (fast inference)
- NVIDIA GPUs (optional but recommended)

**Databases:**
- DuckDB (local analytics)
- Chroma (vector storage)

**ML:**
- HuggingFace Transformers
- PEFT (parameter-efficient fine-tuning)
- Sentence-Transformers (embeddings)

## Performance Considerations

1. **Context Window Management**
   - Automatic truncation when prompts exceed `max_model_len`
   - Reserve space for generation tokens (default: 2048)

2. **Batch Size Tuning**
   - `max_num_batched_tokens`: 65536 (adjust for VRAM)
   - `batch_size`: 32 (adjust for memory)

3. **Concurrency**
   - Ray workers: typically 1 per GPU
   - vLLM concurrency: set to GPU count
   - Asyncio semaphores: limit CPU operations

4. **Memory Optimization**
   - Chunked prefill for long sequences
   - P2P disabled by default (set `NCCL_P2P_DISABLE=1`)
   - Tensor parallelism for large models

## Key Metrics & Statistics

Tracked throughout the pipeline:
- **Parse Rate**: % of theorems successfully parsed
- **Compilation Rate**: % of generated proofs that compile
- **Correctness Rate**: % of compiled proofs that are semantically correct
- **Improvement Rate**: % achieving improvement in target metric
- **Average Improvement**: Mean delta for successful improvements
- **Best-of-N Selection**: Highest improvement achievable with N samples

