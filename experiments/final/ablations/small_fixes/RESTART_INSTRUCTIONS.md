# First Iteration Sweep - Restart Instructions

## Current Status

✅ **Completed:**
- Phase 1: Base data generation
- Phase 2: Training dataset generation (13 datasets)
- Phase 3A: SFT training (4 models: lr1e-06, lr5e-06, lr1e-05, lr2e-05)
- Phase 3B: wSFT LR sweep (4 models: lr1e-06, lr5e-06, lr1e-05, lr2e-05)
- Phase 3C: wSFT data sweep (3 models: vt0.5, vt0.8, vt1.0)
- Phase 4: Model evaluations (7 completed, 4 need to be run)
  - ✅ SFT: lr1e-06, lr5e-06, lr1e-05, lr2e-05
  - ✅ wSFT: lr1e-06, lr5e-06, lr1e-05
  - ❌ wSFT: lr2e-05, vt0.5, vt0.8, vt1.0 (failed due to NFS cache issue)

**Total Models Trained:** 11 models (60GB each merged = ~660GB)

❌ **Not Yet Done:**
- Phase 4: 4 remaining evaluations
- Phase 5: Select best SFT and wSFT models
- Phase 6A: IRPO training sweeps
- Phase 6B: IRPO data sweeps
- Phase 7: DPO training sweeps
- Phase 8: Evaluate IRPO/DPO models
- Phase 9: Select best IRPO/DPO models
- Phase 10: Final analysis

---

## What Was Done

### 1. Fixed NFS Cache Issue
Cleared stale file handles that caused Ray/vLLM timeout:
```bash
rm -rf /home/riyaza/.triton/autotune/*
rm -rf /home/riyaza/.cache/vllm/torch_compile_cache/*
```

### 2. Created Missing Evaluations Script
Created `run_missing_evals.sh` to run the 4 evaluations that failed:
- wSFT_lr2e-05_test
- wSFT_vt0.5_test
- wSFT_vt0.8_test
- wSFT_vt1.0_test

### 3. Updated Main Script
Commented out completed phases 1-4 in `first_iter_sweep.sh` so it will skip to Phase 5 when you run it again.

---

## Next Steps

### Step 1: Run Missing Evaluations (REQUIRED)

From the improver base directory:
```bash
cd /home/riyaza/eval_improver/improver
bash experiments/final/ablations/first_iter_sweep/run_missing_evals.sh
```

This will take approximately 4-6 hours (1-1.5 hours per evaluation).

**Monitor progress:**
```bash
# Check which evaluations exist
ls -d evals/wSFT_*_test

# Expected output after completion:
# evals/wSFT_lr1e-06_test
# evals/wSFT_lr5e-06_test
# evals/wSFT_lr1e-05_test
# evals/wSFT_lr2e-05_test   ← Should be created
# evals/wSFT_vt0.5_test     ← Should be created
# evals/wSFT_vt0.8_test     ← Should be created
# evals/wSFT_vt1.0_test     ← Should be created
```

### Step 2: Run Remaining Phases

After evaluations complete, run the main script to continue from Phase 5:
```bash
cd /home/riyaza/eval_improver/improver
sbatch experiments/final/ablations/first_iter_sweep/first_iter_sweep.sh
```

The script will now:
1. **Phase 5:** Select best SFT and wSFT models based on evaluations
2. **Phase 6A:** Train IRPO models with different LR/hyperparameters
3. **Phase 6B:** Train IRPO models with different data configurations
4. **Phase 7:** Train DPO models
5. **Phase 8:** Evaluate all IRPO/DPO models
6. **Phase 9:** Select best IRPO/DPO models
7. **Phase 10:** Generate final analysis and comparison

---

## Troubleshooting

### If NFS Cache Errors Return
```bash
rm -rf /home/riyaza/.triton/autotune/*
rm -rf /home/riyaza/.cache/vllm/torch_compile_cache/*
```

### If Disk Quota Errors
Check disk usage:
```bash
df -h /data/user_data/riyaza/
du -sh /data/user_data/riyaza/saved_models/ablations/
```

Clean up LoRA adapters (should already be deleted):
```bash
# Check for remaining *_lora directories
ls -d /data/user_data/riyaza/saved_models/ablations/*_lora 2>/dev/null

# Delete if any exist
rm -rf /data/user_data/riyaza/saved_models/ablations/*_lora
```

### If Evaluations Fail Again
Run individual evaluation commands manually:
```bash
export METRIC=length
export MODELS_DIR=/data/user_data/riyaza/saved_models/ablations

./improver run pipeline \
    --run_id wSFT_lr2e-05_test \
    --annotation --informal --examples 4 \
    --metric $METRIC --prompt_id final_test \
    --split test --model ${MODELS_DIR}/wSFT_lr2e-05 \
    --num_blocks 16 \
    --config experiments/final/test_eval.yaml
```

---

## File Locations

**Scripts:**
- Main script: `experiments/final/ablations/first_iter_sweep/first_iter_sweep.sh`
- Missing evals: `experiments/final/ablations/first_iter_sweep/run_missing_evals.sh`
- This file: `experiments/final/ablations/first_iter_sweep/RESTART_INSTRUCTIONS.md`

**Models:**
- All trained models: `/data/user_data/riyaza/saved_models/ablations/`
- 11 merged models (~15GB each)

**Evaluations:**
- All evaluation results: `/home/riyaza/eval_improver/improver/evals/`

**Logs:**
- Main script logs: `logs/final/ablations/first_iter_sweep.{out,err}`

---

## Summary

**To complete the pipeline:**
1. ✅ Cache cleared
2. ✅ Missing evaluation script created
3. ✅ Main script updated (phases 1-4 commented out)
4. ⏳ Run `run_missing_evals.sh` (4-6 hours)
5. ⏳ Run `first_iter_sweep.sh` via sbatch (will continue from Phase 5)

**Total remaining time estimate:** ~24-48 hours for all remaining phases
