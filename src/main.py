"""Entrypoint: load base model, fine-tune the UNet, compare generations, and optionally upload."""

import logging

import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from loader import get_traindata_loader, retrieve_model  # noqa: E402
from repository import upload_model  # noqa: E402
from trainer import start_fine_tunning  # noqa: E402


def fine_tunning_model(device: str):
    """Run the end-to-end fine-tuning flow: base generation, fine-tune, fine-tuned generation.

    Args:
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
        prompt: Text prompt used to compare base vs. fine-tuned generations.

    Returns:
        The fine-tuned ``UNet2DConditionModel``.
    """
    # 1. Load base model and test generation with user-supplied prompt
    logger.info("Step 1: loading base model and running test generation")
    pipeline = retrieve_model(device)
    # prompt = input("Enter a prompt to test the base model: ")
    # test_image = pipeline(prompt).images[0]
    # test_image.show()

    # 2. Load training data and run fine-tuning
    logger.info("Step 2: loading training data and starting fine-tuning")
    traindata_loader = get_traindata_loader()
    fine_tuned_unet = start_fine_tunning(traindata_loader, device)

    # 3. Test generation again with the fine-tuned unet, same prompt
    logger.info("Step 3: running test generation with fine-tuned UNet")
    prompt = input("Enter a prompt to test the base model: ")
    pipeline.unet = fine_tuned_unet
    fine_tuned_image = pipeline(prompt).images[0]
    fine_tuned_image.show()

    logger.info("Finished fine_tunning_model")
    return fine_tuned_unet


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting fine-tuning run on device: %s", device)
    fine_tuned_unet = fine_tunning_model(device)
    save_model = input("Do you wanna save model (Y/N): ")
    if save_model == "Y":
        logger.info("Uploading fine-tuned model")
        upload_model(fine_tuned_unet)
    logger.info("Fine-tuning run finished")
