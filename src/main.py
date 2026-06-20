import torch
from dotenv import load_dotenv

load_dotenv()

from src.loader import load_model, get_traindata_loader
from src.trainer import start_fine_tunning
from src.repository import upload_model


def fine_tunning_model(device: str, prompt: str):
    # 1. Load base model and test generation with user-supplied prompt
    pipeline = load_model(device)
    test_image = pipeline(prompt).images[0]
    test_image.show()

    # 2. Load training data and run fine-tuning
    traindata_loader = get_traindata_loader()
    fine_tuned_unet = start_fine_tunning(traindata_loader, device)

    # 3. Test generation again with the fine-tuned unet, same prompt
    pipeline.unet = fine_tuned_unet
    fine_tuned_image = pipeline(prompt).images[0]
    fine_tuned_image.show()

    return fine_tuned_unet

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = input("Enter a prompt to test the base model: ")
    fine_tuned_unet = fine_tunning_model(device, prompt)
    save_model = input("Do you wanna save model (Y/N): ")
    if save_model == 'Y':
        upload_model(fine_tuned_unet)
