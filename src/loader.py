import logging
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from image_transformer import to_rgb_image_tensors
import os
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL

logger = logging.getLogger(__name__)

# Cargando el dataset de Libros:
base_model_name = os.getenv("BASE_MODEL_NAME", "CompVis/stable-diffusion-v1-4")
dataset_name = os.getenv("DATASET_NAME", "gigant/oldbookillustrations")
dataset_columns = os.getenv("DATASE_COLUMNS", "1600px,info_alt").split(",")
max_train_samples = int(os.getenv("MAX_TRAIN_SAMPLES", 500))
batch_size = int(os.getenv("TRAIN_BATCH_SIZE", 6))
final_model_name = os.getenv("FTUNNING_MODEL_NAME", "vintage-bookshop-practice")
hggf_username = os.getenv("HGGF_USERNAME", "")
hggf_token = os.getenv("HGGF_TOKEN", "")

class Text2ImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        logger.info("Executing Text2ImageDataset.__getitem__")
        example = self.dataset[idx]
        image = to_rgb_image_tensors(example)
        token = tokenizer(example["text"], padding="max_length", truncation=True,
                          max_length=tokenizer.model_max_length, return_tensors="pt")
        item = {
            "pixel_values": image,
            "input_ids": token.input_ids.squeeze(0),
            "attention_mask": token.attention_mask.squeeze(0),
        }
        logger.info("Finished Text2ImageDataset.__getitem__")
        return item

def __load_dataset(split_dataset="train"):
    logger.info("Executing __load_dataset")
    dataset = load_dataset(dataset_name, split=split_dataset).select_columns(dataset_columns)

    selected = dataset.select(range(max_train_samples))
    logger.info("Finished __load_dataset")
    return selected

def load_model(device="cpu"):
    logger.info("Executing load_model")
    # Cargamos un modelo pre-entrenado Stable Diffusion:
    pipeline = StableDiffusionPipeline.from_pretrained(base_model_name).to(device)
    logger.info("Finished load_model")
    return pipeline

def load_finetunned_model(device="cpu"):
    logger.info("Executing load_finetunned_model")
    finetuned_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path=f"{hggf_username}/{final_model_name}-unet",
        token=hggf_token)
    finetuned_unet.to(device)

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_name,
        unet=finetuned_unet
    ).to(device)
    logger.info("Finished load_finetunned_model")
    return pipeline

def get_traindata_loader():
    logger.info("Executing get_traindata_loader")
    filtered_dataset = __load_dataset()

    train_dataset = Text2ImageDataset(filtered_dataset)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info("Finished get_traindata_loader")
    return loader

def retrieve_model_parts(device="cpu"):
    logger.info("Executing retrieve_model_parts")
    # Tokenizador:
    tokenizer = CLIPTokenizer.from_pretrained(base_model_name, subfolder='tokenizer')
    # Scheduler:
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_name, subfolder="scheduler")
    # Text Encoder (CLIP):
    text_encoder = CLIPTextModel.from_pretrained(base_model_name,subfolder="text_encoder").to(device)
    # VAE: Autoencoder:
    vae = AutoencoderKL.from_pretrained(base_model_name, subfolder="vae").to(device)
    # La UNet:
    unet = UNet2DConditionModel.from_pretrained(base_model_name,subfolder="unet").to(device)
    # Congelamos los pesos del VAE y del Text Encoder, ya que solo queremos finetunear la UNet:
    vae.eval()
    text_encoder.eval()

    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    logger.info("Finished retrieve_model_parts")
    return tokenizer, noise_scheduler, text_encoder, vae, unet