"""Dataset and model loading utilities for the vintage-book Stable Diffusion fine-tuning flow."""

import logging
import os

from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

from image_transformer import to_rgb_image_tensors
from repository import BASE_MODEL_LOCAL_PATH, save_model_locally

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
resolution = int(os.getenv("IMAGE_RESOLUTION", 512))
crop_type = os.getenv("IMAGE_CROP_METHOD", "center")

class Text2ImageDataset(Dataset):
    """Wraps a Hugging Face dataset of (image, text) pairs for text-to-image fine-tuning."""

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = to_rgb_image_tensors(
            sample_image=example, 
            column_name=dataset_columns[0], 
            resolution=resolution,
            crop=crop_type)
        token = self.tokenizer(
            example[dataset_columns[1]],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        item = {
            "pixel_values": image,
            "input_ids": token.input_ids.squeeze(0),
            "attention_mask": token.attention_mask.squeeze(0),
        }
        return item


def __load_dataset(split_dataset="train"):
    logger.info("Executing __load_dataset")
    dataset = load_dataset(dataset_name, split=split_dataset).select_columns(dataset_columns)

    selected = dataset.select(range(max_train_samples))
    logger.info("Finished __load_dataset")
    return selected


def __local_model_exists():
    return BASE_MODEL_LOCAL_PATH.exists() and any(
        not entry.name.startswith(".") for entry in BASE_MODEL_LOCAL_PATH.iterdir()
    )


def __load_model(device="cpu"):
    logger.info("Executing __load_model")
    # Cargamos un modelo pre-entrenado Stable Diffusion:
    pipeline = StableDiffusionPipeline.from_pretrained(base_model_name).to(device)
    logger.info("Finished __load_model")
    return pipeline


def retrieve_model(device="cpu"):
    """Return the base Stable Diffusion pipeline, using a local cache when available.

    Checks ``BASE_MODEL_LOCAL_PATH`` first; if the model was already downloaded
    and saved there, it is loaded from disk. Otherwise it is downloaded from
    the Hugging Face Hub and cached locally via ``save_model_locally`` for
    subsequent runs.

    Args:
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        A ``StableDiffusionPipeline`` moved to ``device``.
    """
    logger.info("Executing retrieve_model")
    if __local_model_exists():
        logger.info("Local base model found at %s, loading from disk", BASE_MODEL_LOCAL_PATH)
        pipeline = StableDiffusionPipeline.from_pretrained(BASE_MODEL_LOCAL_PATH).to(device)
    else:
        logger.info("Local base model not found, downloading from Hub")
        pipeline = __load_model(device)
        save_model_locally(pipeline)
    logger.info("Finished retrieve_model")
    return pipeline


def load_finetunned_model(device="cpu"):
    """Download the fine-tuned UNet from the Hub and assemble a full pipeline.

    Args:
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        A ``StableDiffusionPipeline`` using the base model with the
        fine-tuned UNet swapped in, moved to ``device``.
    """
    logger.info("Executing load_finetunned_model")
    finetuned_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path=f"{hggf_username}/{final_model_name}-unet",
        token=hggf_token,
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_name,
        unet=finetuned_unet,
    ).to(device)
    logger.info("Finished load_finetunned_model")
    return pipeline


def get_traindata_loader(tokenizer):
    """Build the training ``DataLoader`` over the vintage-book illustrations dataset.

    Returns:
        A ``torch.utils.data.DataLoader`` yielding batches of
        ``pixel_values``, ``input_ids`` and ``attention_mask``.
    """
    logger.info("Executing get_traindata_loader")
    filtered_dataset = __load_dataset()

    train_dataset = Text2ImageDataset(filtered_dataset, tokenizer)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    logger.info("Finished get_traindata_loader")

    return loader


def retrieve_model_parts(device="cpu"):
    """Load the individual Stable Diffusion components needed for UNet fine-tuning.

    Components are loaded from ``BASE_MODEL_LOCAL_PATH`` when a local cache of
    the base model exists, falling back to the Hugging Face Hub otherwise.
    The VAE and text encoder are frozen (``eval`` mode, ``requires_grad=False``)
    since only the UNet is meant to be fine-tuned.

    Args:
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        Tuple of ``(tokenizer, noise_scheduler, text_encoder, vae, unet)``.
    """
    logger.info("Executing retrieve_model_parts")
    if __local_model_exists():
        logger.info("Local base model found at %s, loading parts from disk", BASE_MODEL_LOCAL_PATH)
        model_source = BASE_MODEL_LOCAL_PATH
    else:
        logger.info("Local base model not found, loading parts from Hub")
        model_source = base_model_name

    # Tokenizador:
    tokenizer = CLIPTokenizer.from_pretrained(model_source, subfolder="tokenizer")
    # Scheduler:
    noise_scheduler = DDPMScheduler.from_pretrained(model_source, subfolder="scheduler")
    # Text Encoder (CLIP):
    text_encoder = CLIPTextModel.from_pretrained(
        model_source, subfolder="text_encoder"
    ).to(device)
    # VAE: Autoencoder:
    vae = AutoencoderKL.from_pretrained(model_source, subfolder="vae").to(device)
    # La UNet:
    unet = UNet2DConditionModel.from_pretrained(model_source, subfolder="unet").to(device)
    # Congelamos los pesos del VAE y del Text Encoder, ya que solo queremos finetunear la UNet:
    vae.eval()
    text_encoder.eval()

    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    logger.info("Finished retrieve_model_parts")
    return tokenizer, noise_scheduler, text_encoder, vae, unet
