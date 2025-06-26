from huggingface_hub import HfApi, HfFolder, Repository, create_repo, login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


# Get the Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

login(token=HF_TOKEN)

repo_name = "riyazahuja/coder_completion_informal"

# create_repo(repo_name, private=False)

model_folder = "/data/user_data/riyaza/saved_models/coder-completion_informal"


tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = AutoModelForCausalLM.from_pretrained(model_folder)
print("Model and tokenizer loaded successfully.")
model.push_to_hub(repo_name, token=HF_TOKEN)
print("Model pushed to Hugging Face Hub successfully.")
tokenizer.push_to_hub(repo_name, token=HF_TOKEN)
print("Tokenizer pushed to Hugging Face Hub successfully.")


api = HfApi()

# api.delete_file(path_in_repo='README.md',repo_id=repo_name, repo_type="model")
# api.upload_file(
#     path_or_fileobj=f"{model_folder}/README.md",
#     path_in_repo="README.md",
#     repo_id=repo_name,
#     repo_type="model",
# )
