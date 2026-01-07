from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import argparse


def merge_models(ref, adapter, output, safe_serialization=True):

    # 1️⃣ Load your base model (8-bit, bf16, etc. as desired)
    base = AutoModelForCausalLM.from_pretrained(
        ref,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2️⃣ Load your LoRA adapter on top
    model = PeftModel.from_pretrained(base, adapter)  # adapter path

    merged = model.merge_and_unload()  # now a plain AutoModelForCausalLM

    merged.save_pretrained(output, safe_serialization=safe_serialization)

    # load the original tokenizer
    tok = AutoTokenizer.from_pretrained(ref)
    # write it into your merged checkpoint folder
    tok.save_pretrained(output)


# adapters = [
#     f"/data/user_data/riyaza/saved_models/wSFT_{t}_{s}"
#     for t in ["replace"]
#     for s in ["big", "small"]
# ]
# ref = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


# for adapter in adapters:
#     output = adapter.replace("wSFT_", "wSFT_merged_")
#     merge_models(ref, adapter, output)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PEFT adapter with base model")
    parser.add_argument(
        "--ref", type=str, required=True, help="Path to reference/base model"
    )
    parser.add_argument(
        "--adapter", type=str, required=True, help="Path to PEFT adapter"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for merged model"
    )
    parser.add_argument(
        "--safe-serialization", action="store_true", default=False,
        help="Use safe serialization (safetensors format)"
    )

    args = parser.parse_args()

    merge_models(args.ref, args.adapter, args.output, args.safe_serialization)


# models = ["wSFT_replace_i2", "wSFT_join_i2", "wSFT_none_i2"]

# for model in models:
#     adapter = f"/data/user_data/riyaza/saved_models/{model}_lora"
#     ref = f"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#     output = f"/data/user_data/riyaza/saved_models/{model}"
#     merge_models(ref, adapter, output)
