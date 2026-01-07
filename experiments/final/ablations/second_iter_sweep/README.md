# Second Iteration Ablation Study (with wSFT)

Comprehensive ablation study for the second iteration of ImProver training, following the full pipeline: **Base Model → wSFT → IRPO**

## Overview

This system automates the complete second iteration ablation pipeline:
1. Run inference on training set with best IRPO model from iteration 1
2. Generate wSFT training datasets with different replay buffer configurations
3. Train 8 wSFT models with different base model initializations
4. Merge LoRA adapters for wSFT models
5. Evaluate wSFT models
6. For each wSFT model, run inference and generate IRPO training data
7. Train 8 IRPO models (one for each wSFT)
8. Evaluate IRPO models
9. Select best IRPO model for potential third iteration

## Key Research Questions

**Base Model Initialization:** Does starting from the best first-iteration model (IRPO_w4_l4) improve performance compared to starting from the base DeepSeek-7B model?

**Replay Buffer Strategy:** How does mixing old training data (from the base run) with new training data affect performance at both the wSFT and IRPO stages? What is the optimal replay ratio?

**Pipeline Effectiveness:** Does the full Base → wSFT → IRPO pipeline improve performance in the second iteration?

## Directory Structure

```
second_iter_sweep/
├── configs/base/
│   ├── wSFT_deepseek_base.yaml  # wSFT with DeepSeek-7B initialization
│   ├── wSFT_iter1_base.yaml     # wSFT with IRPO_w4_l4 initialization
│   └── IRPO_base.yaml            # IRPO base config (uses wSFT models)
├── sweeps/
│   ├── wSFT_sweep.yaml          # 8-way wSFT sweep
│   └── IRPO_sweep.yaml          # 8-way IRPO sweep
├── data/
│   ├── wSFT_norep/              # wSFT data, no replay (arrow format)
│   ├── wSFT_rep0.2/             # wSFT data, 20% replay
│   ├── wSFT_rep0.4/             # wSFT data, 40% replay
│   ├── wSFT_rep0.6/             # wSFT data, 60% replay
│   ├── IRPO_deepseek_norep.jsonl  # IRPO data for each wSFT model
│   ├── IRPO_deepseek_rep0.2.jsonl
│   └── ... (8 IRPO datasets total)
├── scripts/
│   ├── generate_ablation_configs.py  # Config/sweep generator
│   ├── generate_wsft_data.sh         # wSFT data generation
│   ├── generate_irpo_data.sh         # IRPO data generation
│   └── select_best_model.py          # Best model selector
└── second_iter_sweep.sh              # Main orchestration script (10 phases)
```

## Complete Pipeline

### Stage 1: wSFT Training

**Input:** Inference results from IRPO_w4_l4 (`second_iter_base_train`)

**Data Generation:**
- 4 wSFT datasets with different replay buffer configs
- Fixed params: vt=0.5, lr=2e-5, tau=0.8, epsilon=0.1, hardness=0.8

**Training:**
- 8 wSFT models (2 base models × 4 replay configs)
- Base models: DeepSeek-7B, IRPO_w4_l4
- LoRA training with r=64, alpha=128

**Output:** 8 merged wSFT models + evaluations

### Stage 2: IRPO Training

**Input:** Inference results from each of 8 wSFT models

**Data Generation:**
- 8 IRPO datasets (one per wSFT model)
- Fixed params: W=4, L=4, min_gap=1, hardness=0.8

**Training:**
- 8 IRPO models (one for each wSFT)
- Fixed params: beta=0.1, alpha=0.5, lr=5e-6

**Output:** 8 IRPO models + evaluations + best model selection

## Fixed Hyperparameters

### wSFT (from first iteration best)
- **variance_threshold**: 0.5
- **learning_rate**: 2e-5
- **tau**: 0.8
- **epsilon**: 0.1
- **filter_threshold**: 0.8 (hardness)

### IRPO (from first iteration best)
- **rl_beta**: 0.1
- **rpo_alpha**: 0.5
- **learning_rate**: 5e-6
- **max_champions (W)**: 4
- **num_invalid (L)**: 4
- **min_gap**: 1
- **filter_threshold**: 0.8 (hardness)

## Ablation Variables

### Base Model (2 options)
1. **DeepSeek-7B**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
   - Starting from scratch with base model
2. **Iter1 IRPO**: `/data/user_data/riyaza/saved_models/ablations/IRPO_w4_l4`
   - Starting from best first iteration model (warm start)

### Replay Buffer (4 options)
1. **None**: No replay buffer, only new training data
2. **Replace 20%**: Replace 20% of new data with data from `base_length_train`
3. **Replace 40%**: Replace 40% of new data with replay data
4. **Replace 60%**: Replace 60% of new data with replay data

**Total configurations: 2 × 4 = 8 per stage**  
**Total models: 16 (8 wSFT + 8 IRPO)**

## Model Naming Convention

### wSFT Models
- Format: `wSFT_{base}_{replay}`
- Examples: `wSFT_deepseek_norep`, `wSFT_iter1_rep0.2`

### IRPO Models
- Format: `IRPO_{base}_{replay}`
- Examples: `IRPO_deepseek_norep`, `IRPO_iter1_rep0.4`
- Each IRPO model uses the corresponding wSFT model as base

## Quick Start

### 1. Generate Configurations

```bash
cd experiments/final/ablations/second_iter_sweep
python scripts/generate_ablation_configs.py
```

This creates:
- 2 wSFT base configuration files
- 1 wSFT sweep YAML (8 configurations)
- 1 IRPO base configuration file
- 1 IRPO sweep YAML (8 configurations)
- wSFT and IRPO data generation scripts

### 2. Run Ablation Study

Submit the main script to SLURM:

```bash
sbatch second_iter_sweep.sh
```

Or run locally (not recommended due to ~60-90 hour runtime):

```bash
bash second_iter_sweep.sh
```

### 3. Monitor Progress

Check logs:
```bash
tail -f logs/final/ablations/second_iter_sweep.out
tail -f logs/final/ablations/second_iter_sweep.err
```

Monitor WandB:
- wSFT projects: `second_iter_wSFT_{base}_{replay}`
- IRPO projects: `second_iter_IRPO_{base}_{replay}`

## Execution Phases

The main script runs in 10 phases:

### Phase 1: Inference with Best Iter 1 Model
- Runs ImProver pipeline with IRPO_w4_l4
- Metric: length optimization
- Dataset: training split
- Output: `run_id=second_iter_base_train`

### Phase 2: Generate wSFT Datasets
- Creates 4 wSFT datasets with different replay configs
- Preprocesses to arrow format
- Uses data from `second_iter_base_train` + optional replay from `base_length_train`

### Phase 3: Train wSFT Models
- Trains 8 wSFT models using Axolotl sweep
- 2 base model initializations × 4 replay configs
- LoRA training with r=64, alpha=128

### Phase 4: Merge LoRA Adapters
- Merges each wSFT LoRA adapter with its base model
- Creates 8 full wSFT models ready for evaluation and IRPO training

### Phase 5: Evaluate wSFT Models
- Evaluates all 8 wSFT models on test set
- Uses: annotation, informal, 4 examples
- Metric: length

### Phase 6: Select Best wSFT (Informational)
- Identifies best wSFT model based on improvement score
- Saved to `.best_wsft_model`
- Informational only; all 8 continue to IRPO stage

### Phase 7: Generate IRPO Data
- For each wSFT model:
  - Run inference on training set
  - Generate IRPO dataset from that inference run
- Creates 8 IRPO datasets

### Phase 8: Train IRPO Models
- Trains 8 IRPO models using Axolotl sweep
- Each uses corresponding wSFT model as base
- Fixed hyperparams from first iteration best

### Phase 9: Evaluate IRPO Models
- Evaluates all 8 IRPO models on test set
- Uses: annotation, informal, 4 examples
- Metric: length

### Phase 10: Select Best IRPO Model
- Parses all IRPO evaluation results
- Selects model with best (most negative) improvement score
- Saves path to `.best_second_iter_model`

## Expected Timeline

Based on first iteration experience:

- **Phase 1 (Iter1 Inference)**: ~8-12 hours
  - Training set inference with 512 blocks
- **Phase 2 (wSFT data gen)**: ~1 hour
  - 4 datasets + preprocessing
- **Phase 3 (wSFT training)**: ~16-24 hours
  - 8 models, ~2-3 hours each, parallel via Axolotl
- **Phase 4 (Merge)**: ~1 hour
  - LoRA merging for 8 models
- **Phase 5 (wSFT eval)**: ~4-6 hours
  - 8 test set evaluations
- **Phase 6 (Selection)**: <1 minute
- **Phase 7 (IRPO data gen)**: ~12-16 hours
  - 8 inference runs + data generation
- **Phase 8 (IRPO training)**: ~16-24 hours
  - 8 models, parallel training
- **Phase 9 (IRPO eval)**: ~4-6 hours
  - 8 test set evaluations
- **Phase 10 (Selection)**: <1 minute

**Total: ~60-90 hours**

## Output Files

### Generated Files
- `.best_wsft_model` - Path to best wSFT model
- `.best_second_iter_model` - Path to best IRPO model

### Logs
- `logs/final/ablations/second_iter_sweep.out` - Standard output
- `logs/final/ablations/second_iter_sweep.err` - Error output
- WandB logs for each training run (16 total)

### Models
- `/data/user_data/riyaza/saved_models/ablations/second_iter/wSFT_{base}_{replay}` - wSFT models
- `/data/user_data/riyaza/saved_models/ablations/second_iter/IRPO_{base}_{replay}` - IRPO models

### Evaluations
- `evals/wSFT_{base}_{replay}_test/analysis/BoN/data.csv` - wSFT evaluations
- `evals/IRPO_{base}_{replay}_test/analysis/BoN/data.csv` - IRPO evaluations

### Training Data
- `data/wSFT_{suffix}/` - wSFT datasets (arrow format)
- `data/IRPO_{base}_{replay}.jsonl` - IRPO datasets

## Analyzing Results

### View Improvement Scores

```bash
# wSFT models
for model in deepseek_norep deepseek_rep0.2 deepseek_rep0.4 deepseek_rep0.6 \
             iter1_norep iter1_rep0.2 iter1_rep0.4 iter1_rep0.6; do
    echo -n "wSFT_$model: "
    tail -1 evals/wSFT_${model}_test/analysis/BoN/data.csv | cut -d',' -f2
done

# IRPO models
for model in deepseek_norep deepseek_rep0.2 deepseek_rep0.4 deepseek_rep0.6 \
             iter1_norep iter1_rep0.2 iter1_rep0.4 iter1_rep0.6; do
    echo -n "IRPO_$model: "
    tail -1 evals/IRPO_${model}_test/analysis/BoN/data.csv | cut -d',' -f2
done
```

### Check Best Models

```bash
cat .best_wsft_model
cat .best_second_iter_model
```

### WandB Analysis

Compare training curves for:
- wSFT models across base initialization and replay configs
- IRPO models across wSFT starting points
- Training loss, validation metrics, convergence

## Troubleshooting

### Phase 1 Fails (Iter1 Inference)
```bash
# Verify model exists
ls -la /data/user_data/riyaza/saved_models/ablations/IRPO_w4_l4

# Run manually
./improver run pipeline \
    --run_id second_iter_base_train \
    --annotation --informal --examples 4 \
    --metric length --prompt_id final_train \
    --split train --model /data/user_data/riyaza/saved_models/ablations/IRPO_w4_l4 \
    --num_blocks 512 \
    --config experiments/final/test_eval.yaml
```

### Phase 2 Fails (wSFT Data Gen)
```bash
# Run data generation manually
bash scripts/generate_wsft_data.sh

# Or generate single dataset
./improver run training_data \
    --run_id second_iter_base_train \
    --output_path data/wSFT_norep.jsonl \
    --type weighted_sft \
    --tau 0.8 --epsilon 0.1 \
    --variance_threshold 0.5 --filter_threshold 0.8

# Preprocess
python experiments/final/preprocess_weights.py data/wSFT_norep.jsonl data/wSFT_norep
```

### Phase 3 Fails (wSFT Training)
```bash
# Check configs exist
ls -la configs/base/wSFT_*.yaml sweeps/wSFT_sweep.yaml

# Check data exists
ls -la data/wSFT_*/

# Train single model for debugging
axolotl train configs/base/wSFT_deepseek_base.yaml
```

### Phase 7 Fails (IRPO Data Gen)
```bash
# Verify wSFT models exist and were merged
ls -la /data/user_data/riyaza/saved_models/ablations/second_iter/wSFT_*

# Manually run inference + data gen for one model
./improver run pipeline \
    --run_id wSFT_deepseek_norep_train \
    --annotation --informal --examples 4 \
    --metric length --prompt_id final_train \
    --split train \
    --model /data/user_data/riyaza/saved_models/ablations/second_iter/wSFT_deepseek_norep \
    --num_blocks 512 \
    --config experiments/final/test_eval.yaml

./improver run training_data \
    --run_id wSFT_deepseek_norep_train \
    --output_path data/IRPO_deepseek_norep.jsonl \
    --type dpo \
    --max_champions 4 --num_invalid 4 --min_gap 1 --filter_threshold 0.8
```

## Comparison with First Iteration

| Aspect | First Iteration | Second Iteration |
|--------|----------------|------------------|
| **Pipeline** | SFT, wSFT, IRPO, DPO separately | Base → wSFT → IRPO sequentially |
| **Focus** | Hyperparameter search | Base model & replay buffer strategy |
| **Base Models** | DeepSeek-7B only | DeepSeek-7B vs IRPO_w4_l4 |
| **Configs** | ~30+ training runs | 16 training runs (8 wSFT + 8 IRPO) |
| **Hyperparameters** | Sweeping many params | Fixed at best values from iter 1 |
| **Data Source** | Base model inference only | Base + optional replay buffer |
| **Training Stages** | Independent | Sequential (wSFT then IRPO) |

## Next Steps

After the ablation study completes:

### 1. Analyze Results

```bash
# View best models
cat .best_wsft_model
cat .best_second_iter_model

# Compare all IRPO models
python scripts/select_best_model.py \
    IRPO_deepseek_norep_test IRPO_deepseek_rep0.2_test IRPO_deepseek_rep0.4_test IRPO_deepseek_rep0.6_test \
    IRPO_iter1_norep_test IRPO_iter1_rep0.2_test IRPO_iter1_rep0.4_test IRPO_iter1_rep0.6_test \
    --evals-path evals \
    --models-path /data/user_data/riyaza/saved_models/ablations/second_iter
```

### 2. Determine Key Insights

- Does warm-starting from iter1 model help at the wSFT stage?
- Does warm-starting help at the IRPO stage?
- What is the optimal replay buffer ratio for wSFT?
- What is the optimal replay buffer ratio for IRPO?
- Are there interaction effects between base model, replay ratio, and training stage?

### 3. Plan Third Iteration (if needed)

Based on results, decide:
- Use best second iter IRPO model for third iteration
- Adjust replay buffer strategy based on findings
- Consider other ablations (e.g., curriculum learning, data quality filtering)

## Dependencies

- Axolotl (with sweep functionality and weightedSFT plugin)
- PyTorch + DeepSpeed
- ImProver (with pipeline and training_data commands)
- pandas (for select_best_model.py)
- HuggingFace transformers & datasets

## Authors

Generated using Claude Code for the ImProver project.
