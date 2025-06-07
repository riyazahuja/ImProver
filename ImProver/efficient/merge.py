from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# 1️⃣ Load your base model (8-bit, bf16, etc. as desired)
base = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-Prover-V2-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2️⃣ Load your LoRA adapter on top
model = PeftModel.from_pretrained(base, "/data/user_data/riyaza/saved_models/DS2_lora_0")  # adapter path

merged = model.merge_and_unload()  # now a plain AutoModelForCausalLM

merged.save_pretrained("/data/user_data/riyaza/saved_models/DS2_lora_0_merged")

from transformers import AutoTokenizer

# load the original tokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Prover-V2-7B")
# write it into your merged checkpoint folder
tok.save_pretrained("/data/user_data/riyaza/saved_models/DS2_lora_0_merged")