import torch
from dotenv import load_dotenv

load_dotenv()

from src.loader import load_finetunned_model


def test_finetunned_model(device: str, prompt: str):
    # 1. Download the fine-tuned UNet from the Hub and build the pipeline
    pipeline = load_finetunned_model(device)

    # 2. Generate an image from the prompt
    image = pipeline(prompt).images[0]

    # 3. Show the image
    image.show()

    return image


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = input("Enter a prompt to test the fine-tuned model: ")
    test_finetunned_model(device, prompt)