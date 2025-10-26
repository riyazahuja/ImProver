# ImProver: Quick Start Guide for New Developers

## Understanding the System in 5 Minutes

ImProver automates Lean 4 proof optimization through an LLM-powered pipeline:

1. **Extract** theorem metadata from Lean code via metaprogramming
2. **Generate** prompts with context, goal states, and examples
3. **Enhance** prompts with similar proofs via RAG (optional)
4. **Generate** optimized proofs using distributed LLM inference
5. **Evaluate** generated proofs for correctness and metric improvement
6. **Analyze** aggregate results and performance

## Essential Components to Know

### Entry Point: `improver_cli.py`
The command-line interface for all operations.

**Main command groups:**
```bash
./improver metrics     # Define/manage metrics
./improver prompts     # Generate prompts from dataset
./improver rag         # Build retrieval-augmented generation cache
./improver run         # Execute inference pipeline
```

### Three Main Python Modules

**ImProver/basic/** - Inference & Evaluation
- `improver.py` - Main orchestrator
- `inference.py` - Ray/vLLM distributed inference
- `eval_improver.py` - Lean-based evaluation
- `analysis.py` - Metrics aggregation

**ImProver/get_prompts/** - Prompt Generation
- `get_prompts.py` - Async orchestration
- `get_prompts.lean` - Lean theorem extraction
- `rag.py` - Retrieval-augmented generation

**ImProver/build/** - RAG Construction
- `preprocess_rag.py` - Declaration metadata
- `informalize.py` - LLM formalization
- `build_db.py` - Vector database construction

### Three Key Lean Components

**TrainingData/** - Metaprogramming foundation
- `Frontend.lean` - Compiles Lean files, extracts CompilationSteps
- `ExtractGoal.lean` - Extracts goal states at each tactic
- `TreeParser.lean` - Parses proof structure from InfoTrees

**metrics/** - Metric implementations
- Each metric lives in its own directory
- `router.lean` - Central dispatcher for scoring functions
- Metrics are pluggable via the router pattern

**lakefile.lean** - Lake build configuration
- Declares Lean executables for metaprogramming tasks
- Imports dependencies (Mathlib, custom libraries)
- Entry point: `lake exe {command}`

## Key Data Structures

### Dataset Format
```json
{
  "train": {
    "Repository": ["path/to/file.lean", { "file": "path", "theorems": ["name"] }]
  },
  "test": { ... },
  "validation": { ... }
}
```
Located in: `data/final_dataset_fixed.json` or custom paths

### Prompt JSON
Generated in `prompts/{prompts_id}/src/Repository/Module/File.json`
Contains: theorem statement, original proof, goal states, context, RAG results

### Evaluation Results
Generated in `evals/{run_id}/evals/Repository/File.json`
Contains: generated proof, success/failure, metric score, delta (improvement)

## Workflow: Running an Optimization

### Quick Start (5 GPU hours for small dataset)

```bash
# Step 1: Generate prompts (30 min, CPU)
./improver prompts get \
  --dataset_path data/final_dataset_fixed_tiny.json \
  --prompts_id "my_test_$(date +%s)" \
  --split train

# Step 2: Run inference (2 hours, 8 GPUs)
python ImProver/basic/improver.py \
  length \
  data/final_dataset_fixed_tiny.json \
  my_test_prompts_id \
  --split train \
  --model "deepseek-ai/DeepSeek-Prover-V2-7B" \
  --gpus 8 \
  --n 3  # Generate 3 variations per theorem

# Step 3: Evaluate (1 hour, CPU)
python ImProver/basic/eval_improver.py run_id

# Step 4: Analyze (5 min, CPU)
python ImProver/basic/analysis.py --run_id run_id
```

**Output:** `evals/{run_id}/analysis/` contains metrics and success statistics

### With RAG (add 2 hours)

```bash
# After Step 1, add RAG building:
./improver rag build \
  --dataset_path data/final_dataset_fixed_tiny.json \
  --rag_id "my_rag_$(date +%s)" \
  --model "taterowney/informal_proof_embedder" \
  --gpus 8

# Then run inference with RAG:
python ImProver/basic/improver.py length ... --rag_id my_rag
```

## Critical Parameters to Adjust

### For Different Dataset Sizes
- **Tiny** (debugging): 5-10 theorems, 10 min inference
- **Small**: 100 theorems, 1 hour inference
- **Medium**: 1000 theorems, 8 hours inference (8 GPUs)
- **Large**: 10000+ theorems, 24+ hours inference

### For Different Hardware
```python
# In inference.py or as CLI args:
--cpus 64              # Match your CPU count
--gpus 8               # GPU count
--num_blocks 64        # 4x GPU count for 2GB VRAM per block
--batch_size 32        # Reduce if OOM
--max_num_batched_tokens 65536  # Reduce if OOM
```

### For Different Inference Quality
```python
# Balance quality vs. speed/cost
--n 1          # Single generation (fast)
--n 5          # Best-of-5 (3x slower, better results)
--n 10         # Best-of-10 (6x slower, best results)

temperature: 0.3       # Deterministic (use for reproducibility)
temperature: 0.7       # Creative (use for diversity)
```

## Metrics: Understanding Each Type

**Length** (minmax: min)
- Counts number of tactics in proof
- Goal: Shorter proofs
- Evaluation: Fast (syntactic count)

**Readability** (minmax: max)
- LLM-evaluated on 6 criteria (9 points total)
- Goal: More readable, structured proofs
- Evaluation: Slow (requires LLM)
- Rubric: Clarity, theorem usage, layout, comments, variable names, automation

**Declarativity** (minmax: max)
- Counts explicit "have" statements
- Goal: More forward-reasoning style
- Evaluation: Fast (syntactic count)

**Dependency** (minmax: min)
- Counts external theorem references
- Goal: More self-contained proofs
- Evaluation: Fast (import analysis)

**Completion** (minmax: max)
- Boolean: proof compiles without "sorry"
- Goal: Full proof generation
- Evaluation: Moderate (Lean compilation)

**Custom Metrics**
```bash
./improver metrics add \
  --name my_metric \
  --system_prompt "Rewrite to..." \
  --minmax min \
  --example_file metrics/my_metric/examples.lean
```

## Key Files to Edit for Common Tasks

### Adding a New Metric
1. Create `metrics/{name}/` directory
2. Write scoring function in `metrics/{name}/{name}.lean`
3. Create examples in `metrics/{name}/examples.lean`
4. Run: `./improver metrics add --name {name} --system_prompt "..." --minmax min`
5. Function automatically added to `metrics/router.lean`

### Modifying Prompt Structure
Edit in `ImProver/get_prompts/get_prompts.py`:
- System prompt templates
- Context injection logic
- Annotation formatting
- RAG integration

### Tweaking Evaluation
Edit in `ImProver/basic/eval_improver.py`:
- Compilation timeout (default: 20 min)
- Error handling
- Delta calculation
- Success criteria

### Custom Inference Configuration
Edit `ImProver/basic/inference.py`:
- Sampling parameters
- Batch processing
- Token truncation logic
- Ray cluster configuration

## Debugging Tips

### Issue: Out of Memory (OOM)
**Solution:**
```bash
--batch_size 16        # Reduce batch size
--num_blocks 128       # Increase partitions
--max_num_batched_tokens 32768  # Reduce token buffer
```

### Issue: Slow Inference
**Check:**
- GPU utilization: `nvidia-smi` should show >80%
- CPU bottleneck: increase `num_blocks`
- Model loading: runs only once, patience on first batch

### Issue: Low Compilation Success Rate
**Causes:**
- Model generating invalid Lean syntax
- Context window too small (truncation cutting off important info)
- Metric-specific constraints too strict

**Debug:**
- Look at generated proofs in `evals/{run_id}/evals/`
- Check compilation errors in JSON
- Adjust prompt templates
- Try higher temperature for more diversity

### Issue: Evaluation Hangs
**Check:**
- Lean compilation timeout (20 min default)
- Infinite loops in generated proofs
- Infinite import dependencies

**Debug:**
- Run one file manually: `lake exe eval_improver Module.Name metric run_id output.json`
- Increase timeout in `eval_improver.py`

## Database Queries for Analysis

Once evaluation completes, query results with DuckDB:

```bash
# Connect to database
python -c "import duckdb; con = duckdb.connect('evals/run_id/data.duckdb'); con.execute('SELECT * FROM run_data LIMIT 10').df()"

# Or use SQL:
duckdb evals/run_id/data.duckdb
> SELECT COUNT(*), AVG(new_score) FROM run_data WHERE new_correct = true;
```

## Common Mistakes to Avoid

1. **Not checking dataset validity** - Verify `data/dataset.json` has correct file paths before starting inference

2. **Too many GPUs for small dataset** - Use `--num_blocks` not `--gpus` to control parallelism when dataset is small

3. **Forgetting RAG preprocessing time** - Informalization takes 2+ hours for large datasets; plan accordingly

4. **OOM after running a while** - Memory leaks in long-running processes; restart Ray between runs

5. **Comparing across different models** - Always use same model + sampling seed for fair comparison

6. **Wrong metric configuration** - `minmax: "min"` means lower is better; verify this matches your goal

## Performance Expectations

| Task | Hardware | Dataset Size | Time |
|------|----------|--------------|------|
| Prompt Generation | 64 CPU | 100 files | 15 min |
| RAG Building | 8 GPU | 100 files | 2 hours |
| Inference | 8 GPU | 100 theorems (n=1) | 30 min |
| Inference | 8 GPU | 100 theorems (n=5) | 2 hours |
| Evaluation | 64 CPU | 100 theorems | 1 hour |
| Analysis | 1 CPU | 100 theorems | 5 min |

## Getting Help

1. **Error in Lean extraction** - Check lakefile.lean imports, run `lake build`
2. **Python dependency issues** - Check imports in main file, install with pip
3. **Dataset format problems** - Validate against `data/final_dataset_fixed.json` structure
4. **Slow performance** - Check resource utilization, Ray diagnostics at `127.0.0.1:8265`
5. **Unexpected results** - Compare prompt JSON against expected format, check model output samples

## Key Repositories for Context

- **Lean Mathlib** - https://github.com/leanprover-community/mathlib4 (imported via lakefile)
- **Training Data** - https://github.com/semorrison/lean-training-data (basis for Frontend.lean)
- **NTP Toolkit** - https://github.com/cmu-l3/ntp-toolkit (proof structure parsing inspiration)

## Next Steps

1. **For immediate testing**: Use `data/final_dataset_fixed_tiny.json` (pre-defined small dataset)
2. **For production runs**: Prepare custom dataset JSON with your target files
3. **For new metrics**: Study `metrics/length/` as template
4. **For RAG enhancement**: Follow RAG building pipeline with custom embedding model
5. **For training models**: Use `training_data.py` to prepare data from ImProver outputs

