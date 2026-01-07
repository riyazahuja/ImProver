# First Iteration Ablation Study

Comprehensive ablation study infrastructure for the first iteration of ImProver training, using Axolotl's sweep functionality for systematic hyperparameter exploration.

## Overview

This system automates the entire ablation study pipeline:
1. Generate base training data from initial model
2. Create training datasets with different parameters
3. Train models with hyperparameter sweeps
4. Evaluate all trained models
5. Automatically select best models based on improvement scores

## Directory Structure

```
first_iter_sweep/
├── configs/
│   └── base/
│       ├── SFT_base.yaml       # Base SFT configuration
│       ├── wSFT_base.yaml      # Base wSFT configuration
│       ├── IRPO_base.yaml      # Base IRPO configuration
│       └── DPO_base.yaml       # Base DPO configuration
├── sweeps/
│   ├── SFT_lr.yaml             # SFT learning rate sweep
│   ├── wSFT_lr.yaml            # wSFT learning rate sweep
│   ├── wSFT_data.yaml          # wSFT data parameter sweep
│   ├── IRPO_lr.yaml            # IRPO learning rate sweep
│   ├── IRPO_params.yaml        # IRPO beta/alpha/W/L/gap sweep
│   └── DPO_params.yaml         # DPO beta sweep
├── data/                       # Generated training datasets
├── scripts/
│   ├── generate_ablation_configs.py  # Config/sweep generator
│   ├── generate_data.sh              # Data generation commands
│   └── select_best_model.py          # Best model selector
└── first_iter_sweep.sh         # Main orchestration script
```

## Ablation Study Details

### Fixed Parameters
- `tau`: 0.8
- `epsilon`: 0.1
- `effective_batch_size`: 8

### Ablations

#### 1. SFT (Supervised Fine-Tuning)
- **Learning Rate**: [1e-6, 5e-6, 1e-5, 2e-5]
- **Dataset**: Standard SFT data (alpaca format)

#### 2. wSFT (Weighted SFT)
- **Learning Rate**: [1e-6, 5e-6, 1e-5, 2e-5]
- **Variance Threshold**: [0.5, 0.8, 1.0]
- **Dataset**: Preprocessed arrow format with weights

#### 3. IRPO (Iterative Relative Policy Optimization)
- **Learning Rate**: [1e-6, 5e-6, 1e-5, 2e-5]
- **Beta**: [0.02, 0.05, 0.1]
- **Alpha**: [0.2, 0.5, 1.0]
- **W/L Pairs**: [(1,1), (2,2), (4,4), (1,4), (2,4), (4,1), (4,2)]
- **Min Gap**: [0, 1, 2]
- **Base Model**: Best wSFT model

#### 4. DPO (Direct Preference Optimization)
- **Beta**: [0.02, 0.05, 0.1]
- **Alpha**: 0.0 (fixed)
- **Base Model**: Best wSFT model

## Quick Start

### 1. Generate Configs

```bash
cd /path/to/first_iter_sweep
python scripts/generate_ablation_configs.py
```

This creates:
- Base configuration files
- Sweep YAML files
- Data generation script

### 2. Run Ablation Study

Submit the main script to SLURM:

```bash
sbatch first_iter_sweep.sh
```

Or run locally (for testing):

```bash
bash first_iter_sweep.sh
```

### 3. Monitor Progress

Check logs:
```bash
tail -f logs/final/ablations/first_iter_sweep.out
tail -f logs/final/ablations/first_iter_sweep.err
```

## Execution Phases

The main script runs in 8 phases:

### Phase 1: Base Data Generation
- Runs ImProver pipeline on train set with base model
- Generates initial trajectory data

### Phase 2: Dataset Generation
- Creates SFT, wSFT, and IRPO/DPO datasets
- Applies different data generation parameters
- Preprocesses wSFT data into arrow format

### Phase 3: SFT/wSFT Training
- **3A**: SFT learning rate sweep
- **3B**: wSFT learning rate sweep
- **3C**: wSFT data parameter sweep
- Merges LoRA adapters with base model

### Phase 4: SFT/wSFT Evaluation
- Evaluates all SFT and wSFT models on test set
- Uses: annotation, informal, 4 examples, no context

### Phase 5: Model Selection
- Parses evaluation CSVs
- Selects best SFT and wSFT models based on improvement score
- Saves paths to `.best_sft_model` and `.best_wsft_model`

### Phase 6: IRPO/DPO Training
- **6A**: IRPO sweeps using best wSFT model
- **6B**: DPO sweeps using best wSFT model

### Phase 7: IRPO/DPO Evaluation
- Evaluates all IRPO and DPO models on test set

### Phase 8: Final Selection
- Selects best IRPO and DPO models
- Outputs summary of best models across all training types

## Model Selection

The `select_best_model.py` script:
- Reads `evals/[run_id]/analysis/BoN/data.csv`
- Extracts improvement score from last row
- Selects model with most negative improvement (best)
- Outputs model path to file

Example usage:
```bash
python scripts/select_best_model.py \
    run_id_1 run_id_2 run_id_3 \
    --evals-path evals \
    --models-path /path/to/saved_models \
    --output-file .best_model
```

## Axolotl Sweep Format

### Independent Variables
Simple syntax for independent parameters:
```yaml
learning_rate: [1e-5, 2e-5]
lora_r: [16, 32]
```

### Dependent Variables
For tying dataset paths to parameters:
```yaml
_:
  - learning_rate: 1e-5
    datasets:
      - path: /path/to/data1
        type: arrow
    output_dir: /path/to/output1
  - learning_rate: 2e-5
    datasets:
      - path: /path/to/data2
        type: arrow
    output_dir: /path/to/output2
```

## Customization

### Change Base Model
Edit `generate_ablation_configs.py`:
```python
BASE_MODEL = "your-model-name"
```

### Add More Hyperparameters
Modify the sweep generation functions in `generate_ablation_configs.py`:
```python
def generate_custom_sweep():
    return {
        "_": [
            {
                "param1": value1,
                "param2": value2,
                # ...
            }
        ]
    }
```

### Change Evaluation Settings
Edit `first_iter_sweep.sh` evaluation commands:
```bash
./improver run pipeline \
    --run_id my_run \
    --annotation --informal --examples 4 \  # Modify these
    --metric length --prompt_id final_test \
    --split test --model $MODEL_PATH \
    --num_blocks 16 \
    --config experiments/final/test_eval.yaml
```

## Troubleshooting

### Config Generation Issues
```bash
# Regenerate all configs
python scripts/generate_ablation_configs.py
```

### Data Generation Failures
```bash
# Run data generation manually
bash scripts/generate_data.sh

# Or run individual commands
./improver run training_data --run_id base_length_train \
    --output_path data/SFT_basic.jsonl --type sft
```

### Evaluation CSV Not Found
- Check that evaluation completed: `ls evals/[run_id]/analysis/BoN/`
- Verify run_id matches: check logs for actual run IDs used
- Ensure pipeline ran successfully: check error logs

### SLURM Out of Memory
Adjust batch sizes in base configs:
```yaml
micro_batch_size: 2  # Reduce this
gradient_accumulation_steps: 8  # Increase to maintain effective BS
```

## Output Files

### Generated Files
- `.best_sft_model` - Path to best SFT model
- `.best_wsft_model` - Path to best wSFT model
- `.best_irpo_model` - Path to best IRPO model
- `.best_dpo_model` - Path to best DPO model

### Logs
- `logs/final/ablations/first_iter_sweep.out` - Standard output
- `logs/final/ablations/first_iter_sweep.err` - Error output
- WandB logs for each training run

### Models
- `/home/riyaz/saved_models/ablations/[model_name]/`

### Evaluations
- `evals/[run_id]/analysis/BoN/data.csv`

## Next Steps

After the ablation study completes:

1. Review best models:
   ```bash
   cat .best_sft_model
   cat .best_wsft_model
   cat .best_irpo_model
   cat .best_dpo_model
   ```

2. Use best models for next iteration:
   - Use best wSFT model as base for IRPO/DPO
   - Use best IRPO/DPO model for next iteration's pipeline

3. Analyze results in detail:
   - Check WandB dashboards
   - Review evaluation CSVs
   - Compare improvement scores across ablations

## Dependencies

- Axolotl (with sweep functionality)
- PyTorch + DeepSpeed
- ImProver (with pipeline and training_data commands)
- pandas (for select_best_model.py)
- HuggingFace transformers & datasets

## Authors

Generated using Claude Code for the ImProver project.
