import logging

import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from loader import load_model, get_traindata_loader
from trainer import start_fine_tunning
from repository import upload_model


def fine_tunning_model(device: str, prompt: str):

    # 1. Load base model and test generation with user-supplied prompt
    logger.info("Step 1: loading base model and running test generation")
    pipeline = load_model(device)
    test_image = pipeline(prompt).images[0]
    test_image.show()

    # 2. Load training data and run fine-tuning
    logger.info("Step 2: loading training data and starting fine-tuning")
    traindata_loader = get_traindata_loader()
    fine_tuned_unet = start_fine_tunning(traindata_loader, device)

    # 3. Test generation again with the fine-tuned unet, same prompt
    logger.info("Step 3: running test generation with fine-tuned UNet")
    pipeline.unet = fine_tuned_unet
    fine_tuned_image = pipeline(prompt).images[0]
    fine_tuned_image.show()

    logger.info("Finished fine_tunning_model")
    return fine_tuned_unet

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting fine-tuning run on device: %s", device)
    prompt = input("Enter a prompt to test the base model: ")
    fine_tuned_unet = fine_tunning_model(device, prompt)
    save_model = input("Do you wanna save model (Y/N): ")
    if save_model == 'Y':
        logger.info("Uploading fine-tuned model")
        upload_model(fine_tuned_unet)
    logger.info("Fine-tuning run finished")
