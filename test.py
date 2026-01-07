import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path


def upload_models_to_huggingface():
    # Initialize Hugging Face API
    api = HfApi()

    # Path to saved models directory
    models_dir = Path("/data/user_data/trowney/saved_models")

    # Check if directory exists
    if not models_dir.exists():
        print(f"Directory {models_dir} does not exist")
        return

    # Iterate through each subdirectory
    for model_path in models_dir.iterdir():
        if model_path.is_dir():
            model_name = model_path.name
            repo_id = f"taterowney/{model_name}"  # Replace 'your_username' with your HF username

            try:
                # Create repository on Hugging Face
                create_repo(repo_id=repo_id, exist_ok=True)
                print(f"Created/verified repository: {repo_id}")

                # Upload all files in the model directory
                api.upload_folder(
                    folder_path=str(model_path), repo_id=repo_id, repo_type="model"
                )
                print(f"Successfully uploaded {model_name} to {repo_id}")

            except Exception as e:
                print(f"Error uploading {model_name}: {str(e)}")


if __name__ == "__main__":
    upload_models_to_huggingface()
