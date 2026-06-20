import logging
import os

logger = logging.getLogger(__name__)

final_model_name = os.getenv("FTUNNING_MODEL_NAME", "vintage-bookshop")
hggf_username = os.getenv("HGGF_USERNAME", "")

def upload_model(unet):
    logger.info("Executing upload_model")

    unet.push_to_hub(
        repo_id=f"{hggf_username}/{final_model_name}-unet",
        commit_message="Fine tunned model upload",
        private=True
    )
    logger.info("Finished upload_model")