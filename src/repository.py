import os

final_model_name = os.getenv("FTUNNING_MODEL_NAME", "vintage-bookshop")
hggf_username = os.getenv("HGGF_USERNAME", "")

def upload_model(unet):

    unet.push_to_hub(
        repo_id=f"{hggf_username}/{final_model_name}-unet",
        commit_message="Fine tunned model upload",
        private=True
    )