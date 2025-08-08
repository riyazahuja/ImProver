from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

ref = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
adapter ="/data/user_data/riyaza/saved_models/r1_qwen7b_32k_post_dpo"

output = "/data/user_data/riyaza/saved_models/r1_qwen7b_32k_post_dpo_full"
 
# 1️⃣ Load your base model (8-bit, bf16, etc. as desired)
base = AutoModelForCausalLM.from_pretrained(
    ref,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2️⃣ Load your LoRA adapter on top
model = PeftModel.from_pretrained(base, adapter)  # adapter path

merged = model.merge_and_unload()  # now a plain AutoModelForCausalLM

merged.save_pretrained(output)

from transformers import AutoTokenizer

# load the original tokenizer
tok = AutoTokenizer.from_pretrained(ref)
# write it into your merged checkpoint folder
tok.save_pretrained(output)