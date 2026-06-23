"""Persistence helpers for Stable Diffusion models: local disk cache and Hugging Face Hub upload."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

final_model_name = os.getenv("FTUNNING_MODEL_NAME", "vintage-bookshop")
hggf_username = os.getenv("HGGF_USERNAME", "")
hggf_token = os.getenv("HGGF_TOKEN", "")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_LOCAL_PATH = PROJECT_ROOT / "model" / "base"


def upload_model(unet):
    """Push a fine-tuned UNet to the Hugging Face Hub as a private repo.

    Args:
        unet: A ``UNet2DConditionModel`` instance with fine-tuned weights.
    """
    logger.info("Executing upload_model")

    if hggf_username == '' or hggf_token == '':
        logger.error("HGGF user name or password are empty")

    else:
        unet.push_to_hub(
            repo_id=f"{hggf_username}/{final_model_name}-unet",
            commit_message="Fine tunned model upload",
            token=hggf_token
        )
        logger.info("Finished upload_model")


def save_model_locally(pipeline):
    """Save a pipeline to the l ocal base-model cache directory.

    Args:
        pipeline: A ``StableDiffusionPipeline`` instance to persist under
            ``BASE_MODEL_LOCAL_PATH``.
    """
    logger.info("Executing save_model_locally")
    BASE_MODEL_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(BASE_MODEL_LOCAL_PATH)
    logger.info("Finished save_model_locally, saved to %s", BASE_MODEL_LOCAL_PATH)
