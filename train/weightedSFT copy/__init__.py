# axolotl/integrations/weightedSFT/__init__.py
import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import Trainer

from axolotl.integrations.base import BasePlugin
from .args import WeightedSFTArgs

# Public collators Axolotl documents
from axolotl.utils.collators.batching import (
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)

LOG = logging.getLogger("axolotl.integrations.weightedSFT")


# ---------- Collator wrappers that forward "weight" ----------
class _WeightedCollatorBase:
    """
    Wraps an Axolotl collator and injects a per-sample 'sample_weight' tensor.
    """

    def __init__(self, inner_collator):
        self.inner = inner_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pop "weight" from each example; default to 1.0 if missing
        weights = [float(f.pop("weight", 1.0)) for f in features]
        batch = self.inner(features)
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        
        import os, sys
        if os.environ.get("W_SFT_DEBUG") == "1":
            print(f"[weighted_sft.collator] weights={weights}", flush=True)
        
        try:
            LOG.debug("[weighted_sft.collator] injected sample_weight tensor shape=%s", batch["sample_weight"].shape)
        except Exception:
            pass
        return batch


class WeightedDataCollatorForSeq2Seq(_WeightedCollatorBase):
    """
    Wrapper for DataCollatorForSeq2Seq (non-packed training).
    Signature mirrors the base collator; we build it internally.
    """
    def __init__(self, tokenizer, **kwargs):
        super().__init__(DataCollatorForSeq2Seq(tokenizer, **kwargs))


class WeightedV2BatchSamplerDataCollatorForSeq2Seq(_WeightedCollatorBase):
    """
    Wrapper for V2BatchSamplerDataCollatorForSeq2Seq (sample packing).
    """
    def __init__(self, tokenizer, **kwargs):
        super().__init__(V2BatchSamplerDataCollatorForSeq2Seq(tokenizer, **kwargs))


# ---------- Trainer that applies per-example weights ----------
class WeightedSFTTrainer(Trainer):
    def __init__(self, *args, normalize_batch_weights: bool = True, **kwargs):
        # Axolotl passes eval_data_collator to its own Trainer subclass; HF Trainer doesn't accept it.
        kwargs.pop("eval_data_collator", None)
        # Older Axolotl versions may also pass bench_data_collator
        kwargs.pop("bench_data_collator", None)
        kwargs.pop("dataset_tags",None)
        super().__init__(*args, **kwargs)
        self.normalize_batch_weights = normalize_batch_weights
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        
        # --- FORCE-WRAP COLLATOR HERE ---
        try:
            # Only wrap once
            if not isinstance(self.data_collator, _WeightedCollatorBase):
                self.data_collator = _WeightedCollatorBase(self.data_collator)
                LOG.info("[weighted_sft] wrapped trainer.data_collator with _WeightedCollatorBase: %s",
                         type(self.data_collator).__name__)
        except Exception as e:
            LOG.warning("[weighted_sft] failed to wrap data_collator: %r", e)
        # --------------------------------
        
        self.last_sample_weight=None
        self.last_per_sample_loss=None

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        LOG.info("[loss] COMPUTING LOSS")
        # Expect standard Axolotl fields + our 'sample_weight'
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        vocab_size = logits.size(-1)
        loss_flat = self.loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1),
        )  # (B*T,)

        loss_tok = loss_flat.view(labels.size(0), -1)  # (B, T)
        # Average per sample over valid tokens (labels == -100 already zeroed)
        valid_counts = (labels != -100).sum(dim=1).clamp_min(1)
        per_sample_loss = (loss_tok.sum(dim=1) / valid_counts)  # (B,)

        self.last_per_sample_loss=per_sample_loss.detach()

        if sample_weight is None:
            loss = per_sample_loss.mean()
            self.last_sample_weight = None
            LOG.info(f"[weighted_sft] SAMPLE WEIGHT IS NONE:\nINPUT: {inputs}\nLABELS: {labels}")
        else:
            w = sample_weight.to(per_sample_loss.device)
            if self.normalize_batch_weights:
                # Sum weights to batch_size to keep loss scale roughly invariant
                w = w * (per_sample_loss.size(0) / w.sum().clamp_min(1e-8))
                
            self.last_sample_weight = w.detach()
            try:
                step = getattr(self.state, "global_step", None)
            except Exception:
                step = None
            msg = (
                f"[weighted_sft] step={step} sum_w={w.sum().item():.4f} "
                f"mean_w={w.mean().item():.4f} per_sample_loss={per_sample_loss.tolist()}"
            )
            # Always log to the Axolotl logger at INFO so it shows up; also print when env var is set
            LOG.info(msg)
            import os
            if os.environ.get("W_SFT_DEBUG") == "1":
                print(msg, flush=True)
            loss = (w * per_sample_loss).mean()

        return (loss, outputs) if return_outputs else loss


# ---------- Plugin wiring ----------
class WeightedSFTPlugin(BasePlugin):
    """
    Axolotl plugin:
      - exposes pydantic args (normalize_batch_weights)
      - swaps in a collator that forwards per-example weights
      - swaps in a Trainer that applies those weights to the loss
    """

    def get_input_args(self):
        # fully-qualified path to the pydantic args model
        return "axolotl.integrations.weightedSFT.args.WeightedSFTArgs"

    def get_trainer_cls(self, cfg):
        LOG.info(
                "[trainer]GETTING TRAINER",

            )
        return WeightedSFTTrainer

    def get_collator_cls_and_kwargs(self, cfg, is_eval: bool = False):
        """
        Choose the appropriate wrapper based on packing flags in cfg.
        Axolotl expects a (collator_cls, kwargs) tuple here — returning only the class
        can lead to the default collator being used and our weights being ignored.
        We also emit a log so it's obvious at runtime which collator is selected.
        """
        LOG.info(f"[collator] ENTERED COLLATOR")
        # Use distinct flags for train vs. eval
        use_packing = bool(cfg.get("eval_sample_packing") if is_eval else cfg.get("sample_packing"))
        collator_cls = (
            WeightedV2BatchSamplerDataCollatorForSeq2Seq if use_packing else WeightedDataCollatorForSeq2Seq
        )
        try:
            LOG.info(
                "[weighted_sft] selecting collator=%s is_eval=%s use_packing=%s",
                collator_cls.__name__, is_eval, use_packing,
            )
        except Exception as e:
            pass
        # IMPORTANT: return (class, kwargs) so the trainer builder instantiates *our* collator
        return collator_cls, {}

    def get_training_args_mixin(self):
        return WeightedSFTArgs

    def get_training_args(self, cfg):
        # forward normalize flag to Trainer.__init__
        return {"normalize_batch_weights": cfg.get("normalize_batch_weights", True)}