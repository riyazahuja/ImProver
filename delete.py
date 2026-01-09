from huggingface_hub import HfApi, list_models
import os


def delete_all_my_models():
    # Initialize the API
    api = HfApi()

    # Get your username (you'll need to be logged in)
    user_info = api.whoami()
    username = user_info["name"]

    # List all your models
    models = list_models(author=username)

    # Delete each model
    for model in models:
        try:
            api.delete_repo(repo_id=model.id, repo_type="model")
            print(f"Deleted: {model.id}")
        except Exception as e:
            print(f"Failed to delete {model.id}: {e}")


if __name__ == "__main__":
    # Make sure you're logged in first: huggingface-cli login
    delete_all_my_models()
