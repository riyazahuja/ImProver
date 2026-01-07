# Second Iteration Ablation Study

Comprehensive ablation study for the second iteration of ImProver training, focusing on base model initialization and replay buffer strategies.

## Overview

This system automates the second iteration ablation pipeline:
1. Run inference on training set with best IRPO model from iteration 1
2. Generate training datasets with different replay buffer configurations
3. Train IRPO models with different base model initializations
4. Evaluate all trained models
5. Select best model for potential third iteration

## Key Research Questions

**Base Model Initialization:** Does starting from the best first-iteration model improve performance compared to starting from the base DeepSeek-7B model?

**Replay Buffer Strategy:** How does mixing old training data (from the base run) with new training data affect performance? What is the optimal replay ratio?

## Directory Structure

```
second_iter_sweep/
├── configs/base/
│   ├── IRPO_deepseek_base.yaml  # IRPO with DeepSeek-7B initialization
│   └── IRPO_iter1_base.yaml     # IRPO with IRPO_w4_l4 initialization
├── sweeps/
│   └── IRPO_sweep.yaml          # 8-way sweep (2 base models × 4 replay configs)
├── data/
│   ├── IRPO_norep.jsonl         # No replay buffer
│   ├── IRPO_rep0.2.jsonl        # Replace 20% with replay data
│   ├── IRPO_rep0.4.jsonl        # Replace 40% with replay data
│   └── IRPO_rep0.6.jsonl        # Replace 60% with replay data
├── scripts/
│   ├── generate_ablation_configs.py  # Config/sweep generator
│   ├── generate_data.sh              # Data generation script
│   └── select_best_model.py          # Best model selector
└── second_iter_sweep.sh              # Main orchestration script
```

## Ablation Study Details

### Fixed Parameters

Determined from first iteration sweep results:

**Training hyperparameters (from best IRPO model):**
- `rl_beta`: 0.1
- `rpo_alpha`: 0.5
- `learning_rate`: 5e-6
- `num_epochs`: 1

**Data generation parameters:**
- `max_champions` (W): 4
- `num_invalid` (L): 4
- `min_gap`: 1
- `filter_threshold`: 0.8 (hardness threshold)

### Ablation Variables

#### Base Model (2 options)
1. **DeepSeek-7B**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
   - Starting from scratch with base model
2. **Iter1 IRPO**: `/data/user_data/riyaza/saved_models/ablations/IRPO_w4_l4`
   - Starting from best first iteration model

#### Replay Buffer (4 options)
1. **None**: No replay buffer, only new training data
2. **Replace 20%**: Replace 20% of new data with data from `base_length_train`
3. **Replace 40%**: Replace 40% of new data with data from `base_length_train`
4. **Replace 60%**: Replace 60% of new data with data from `base_length_train`

**Total configurations: 2 × 4 = 8 training runs**

## Quick Start

### 1. Generate Configurations

```bash
cd experiments/final/ablations/second_iter_sweep
python scripts/generate_ablation_configs.py
```

This creates:
- 2 base IRPO configuration files
- 1 sweep YAML file with 8 configurations
- Data generation script

### 2. Run Ablation Study

Submit the main script to SLURM:

```bash
sbatch second_iter_sweep.sh
```

Or run locally (not recommended due to computational requirements):

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
- Project names follow pattern: `second_iter_{base}_{replay}`
- Example: `second_iter_deepseek_rep0.2`, `second_iter_iter1_norep`

## Execution Phases

The main script runs in 5 phases:

### Phase 1: Inference on Training Set
- Runs ImProver pipeline with best IRPO model from iteration 1 (`IRPO_w4_l4`)
- Metric: length optimization
- Dataset: training split
- Output: `run_id=second_iter_base_train`

### Phase 2: Dataset Generation
- Generates 4 different IRPO training datasets
- All use: W=4, L=4, min_gap=1, hardness=0.8
- Varies replay buffer configuration
- Replay source: `base_length_train` (original base run)

### Phase 3: Model Training
- Trains 8 IRPO models using Axolotl sweep
- 2 base model initializations × 4 replay configs
- Each model saved to `/data/user_data/riyaza/saved_models/ablations/second_iter/{base}_{replay}`

### Phase 4: Model Evaluation
- Evaluates all 8 models on test set
- Uses: annotation, informal, 4 examples
- Metric: length
- Output: evaluation CSVs with improvement scores

### Phase 5: Model Selection
- Parses all evaluation results
- Selects model with best (most negative) improvement score
- Saves path to `.best_second_iter_model`

## Model Naming Convention

Models are named: `{base}_{replay}`

**Examples:**
- `deepseek_norep` - DeepSeek-7B base, no replay
- `deepseek_rep0.2` - DeepSeek-7B base, 20% replay
- `iter1_norep` - IRPO_w4_l4 base, no replay
- `iter1_rep0.6` - IRPO_w4_l4 base, 60% replay

## Output Files

### Generated Files
- `.best_second_iter_model` - Path to best model from second iteration

### Logs
- `logs/final/ablations/second_iter_sweep.out` - Standard output
- `logs/final/ablations/second_iter_sweep.err` - Error output
- WandB logs for each training run (8 total)

### Models
- `/data/user_data/riyaza/saved_models/ablations/second_iter/{model_name}/`

### Evaluations
- `evals/{model_name}_test/analysis/BoN/data.csv`

### Training Data
- `data/IRPO_norep.jsonl` - Generated from `second_iter_base_train`
- `data/IRPO_rep0.2.jsonl` - 80% new + 20% replay
- `data/IRPO_rep0.4.jsonl` - 60% new + 40% replay
- `data/IRPO_rep0.6.jsonl` - 40% new + 60% replay

## Expected Timeline

Based on first iteration experience:

- **Phase 1 (Inference)**: ~8-12 hours
  - Training set inference with 512 blocks
- **Phase 2 (Data Gen)**: ~30 minutes
  - 4 dataset generation commands
- **Phase 3 (Training)**: ~16-24 hours
  - 8 models, ~2-3 hours each
  - Can run in parallel via Axolotl sweep
- **Phase 4 (Evaluation)**: ~4-6 hours
  - 8 test set evaluations, 16 blocks each
- **Phase 5 (Selection)**: ~1 minute

**Total: ~30-40 hours**

## Analyzing Results

### View Improvement Scores

```bash
# For a specific model
cat evals/deepseek_norep_test/analysis/BoN/data.csv

# Compare all models
for model in deepseek_norep deepseek_rep0.2 deepseek_rep0.4 deepseek_rep0.6 \
             iter1_norep iter1_rep0.2 iter1_rep0.4 iter1_rep0.6; do
    echo -n "$model: "
    tail -1 evals/${model}_test/analysis/BoN/data.csv | cut -d',' -f2
done
```

### Check Best Model

```bash
cat .best_second_iter_model
```

### WandB Analysis

Visit WandB projects to compare:
- Training loss curves
- Validation metrics
- Training duration
- Memory usage

## Troubleshooting

### Inference Phase Fails
```bash
# Check model exists
ls -la /data/user_data/riyaza/saved_models/ablations/IRPO_w4_l4

# Manually run inference
./improver run pipeline \
    --run_id second_iter_base_train \
    --annotation --informal --examples 4 \
    --metric length --prompt_id final_train \
    --split train --model /data/user_data/riyaza/saved_models/ablations/IRPO_w4_l4 \
    --num_blocks 512 \
    --config experiments/final/test_eval.yaml
```

### Data Generation Fails
```bash
# Run data generation manually
bash scripts/generate_data.sh

# Or run individual dataset generation
./improver run training_data \
    --run_id second_iter_base_train \
    --output_path data/IRPO_norep.jsonl \
    --type dpo \
    --max_champions 4 --num_invalid 4 \
    --min_gap 1 --filter_threshold 0.8
```

### Training Fails
```bash
# Check DeepSpeed config exists
ls -la deepspeed_configs/zero3_bf16_cpuoffload_all_custom.json

# Check data files exist
ls -la experiments/final/ablations/second_iter_sweep/data/IRPO_*.jsonl

# Manually train a single model (for debugging)
axolotl train configs/base/IRPO_deepseek_base.yaml
```

### Evaluation Fails
```bash
# Manually evaluate a single model
./improver run pipeline \
    --run_id deepseek_norep_test \
    --annotation --informal --examples 4 \
    --metric length --prompt_id final_test \
    --split test --model /data/user_data/riyaza/saved_models/ablations/second_iter/deepseek_norep \
    --num_blocks 16 \
    --config experiments/final/test_eval.yaml
```

## Comparison with First Iteration

| Aspect | First Iteration | Second Iteration |
|--------|----------------|------------------|
| **Focus** | Hyperparameter search (lr, beta, alpha, W, L, gap) | Base model & replay buffer strategy |
| **Base Model** | DeepSeek-7B | DeepSeek-7B vs IRPO_w4_l4 |
| **Training Algorithms** | SFT, wSFT, IRPO, DPO | IRPO only |
| **Configs** | ~30+ training runs | 8 training runs |
| **Hyperparameters** | Sweeping many params | Fixed at best values from iter 1 |
| **Data Source** | Base model inference only | Base + optional replay buffer |

## Next Steps

After the ablation study completes:

### 1. Analyze Results

```bash
# View best model
cat .best_second_iter_model

# Compare all models
python scripts/select_best_model.py \
    deepseek_norep_test deepseek_rep0.2_test deepseek_rep0.4_test deepseek_rep0.6_test \
    iter1_norep_test iter1_rep0.2_test iter1_rep0.4_test iter1_rep0.6_test \
    --evals-path evals \
    --models-path /data/user_data/riyaza/saved_models/ablations/second_iter
```

### 2. Determine Key Insights

- Does warm-starting from iter1 model help?
- What is the optimal replay buffer ratio?
- Are there interaction effects between base model and replay ratio?

### 3. Plan Third Iteration (if needed)

Based on results, decide:
- Use best second iter model for third iteration
- Adjust replay buffer strategy
- Consider other ablations (e.g., different replay sources, curriculum learning)

## Dependencies

- Axolotl (with sweep functionality)
- PyTorch + DeepSpeed
- ImProver (with pipeline and training_data commands)
- pandas (for select_best_model.py)
- HuggingFace transformers & datasets

## Authors

Generated using Claude Code for the ImProver project.
