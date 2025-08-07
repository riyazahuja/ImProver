# my_plugins/weighted_sft.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from transformers import Trainer
from axolotl.integrations.base import BasePlugin
from axolotl.utils.data import DEFAULT_COLLATOR_CLASS  # fallback if needed

# --- Collator that forwards "weight" -> batch["sample_weight"] ---
class WeightedSFTCollator:
    def __init__(self, base_collator_cls, **base_kwargs):
        self.base = base_collator_cls(**base_kwargs)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pop "weight" from each example; default 1.0 if missing
        weights = [float(feat.pop("weight", 1.0)) for feat in features]
        batch = self.base(features)
        # shape: (batch,)
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch

# --- Trainer that applies per-example weights to the loss ---
class WeightedSFTTrainer(Trainer):
    def __init__(self, *args, normalize_batch_weights: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_batch_weights = normalize_batch_weights
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = None):
        """
        Expect inputs to include:
          - input_ids, attention_mask, labels (standard Axolotl SFT)
          - sample_weight: (batch,)
        We compute per-token losses -> mask -> mean per sample -> weight -> mean over batch.
        """
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        # shift for LM loss (Axolotl typically prepares labels already; keep simple):
        vocab_size = logits.size(-1)
        loss_flat = self.loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1),
        )  # (B*T,)

        loss_tok = loss_flat.view(labels.size(0), -1)  # (B, T)

        # ignore padding: labels == -100 do not contribute; CrossEntropyLoss with reduction="none" already gives 0 for -100
        # Average per sample over *valid* tokens:
        valid_counts = (labels != -100).sum(dim=1).clamp_min(1)
        per_sample_loss = (loss_tok.sum(dim=1) / valid_counts)  # (B,)

        if sample_weight is None:
            # fallback: standard mean
            loss = per_sample_loss.mean()
        else:
            # normalize weights so sum == batch_size (keeps LR/scale stable)
            w = sample_weight.to(per_sample_loss.device)
            if self.normalize_batch_weights:
                scale = (per_sample_loss.size(0) / w.sum().clamp_min(1e-8))
                w = w * scale
            loss = (w * per_sample_loss).mean()

        return (loss, outputs) if return_outputs else loss

# --- Plugin wiring ---
@dataclass
class WeightedSFTArgs:
    normalize_batch_weights: bool = True  # exposed in YAML via plugin args if desired

class WeightedSFTPlugin(BasePlugin):
    def get_trainer_cls(self, cfg):
        return WeightedSFTTrainer

    def get_collator_cls_and_kwargs(self, cfg, is_eval: bool = False):
        # Grab Axolotl’s default collator class/kwargs from cfg
        base_cls, base_kwargs = DEFAULT_COLLATOR_CLASS(cfg, is_eval)
        collator_cls = lambda **kw: WeightedSFTCollator(base_cls, **kw)
        return collator_cls, base_kwargs

    def get_training_args_mixin(self):
        return WeightedSFTArgs

    def get_training_args(self, cfg):
        # forward normalize flag down to the Trainer __init__
        return {"normalize_batch_weights": cfg.get("normalize_batch_weights", True)}