"""Standalone script to validate an uploaded fine-tuned UNet by generating an image."""

import logging

import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from loader import load_finetunned_model  # noqa: E402


def test_finetunned_model(device: str, prompt: str):
    """Download the fine-tuned UNet from the Hub and generate a test image.

    Args:
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
        prompt: Text prompt used for generation.

    Returns:
        The generated PIL ``Image``.
    """
    logger.info("Executing test_finetunned_model")

    # 1. Download the fine-tuned UNet from the Hub and build the pipeline
    logger.info("Step 1: downloading fine-tuned UNet and building pipeline")
    pipeline = load_finetunned_model(device)

    # 2. Generate an image from the prompt
    logger.info("Step 2: generating image from prompt")
    image = pipeline(prompt).images[0]

    # 3. Show the image
    logger.info("Step 3: showing generated image")
    image.show()

    logger.info("Finished test_finetunned_model")
    return image


if __name__ == "__main__":
    device = "cpu"
    logger.info("Starting fine-tuned model test on device: %s", device)
    prompt = input("Enter a prompt to test the fine-tuned model: ")
    test_finetunned_model(device, prompt)
    logger.info("Fine-tuned model test finished")
