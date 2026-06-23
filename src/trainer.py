"""Fine-tuning loop for the Stable Diffusion UNet on the vintage-book illustrations dataset."""

import logging
import os

import torch
from accelerate import Accelerator
from tqdm import tqdm

from loader import retrieve_model_parts

logger = logging.getLogger(__name__)

learning_rate = float(os.getenv("LEARNING_RATE", "1e-5"))
training_epochs = int(os.getenv("TRAIN_EPOCHS", "2"))


def __prepare_training(unet, traindata_loader):
    logger.info("Executing __prepare_training")
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # Acelerador:
    accelerator = Accelerator()
    unet, optimizer, __ = accelerator.prepare(unet, optimizer, traindata_loader)
    logger.info("Accelerator device: %s", accelerator.device)

    logger.info("Finished __prepare_training")
    return accelerator, unet, optimizer


def start_fine_tunning(traindata_loader, model_parts: tuple, device: str="cpu"):
    """Fine-tune the Stable Diffusion UNet on the given training data.

    Freezes the VAE and text encoder (via ``retrieve_model_parts``) and trains
    only the UNet to predict the noise added at each diffusion timestep.

    Args:
        traindata_loader: A ``torch.utils.data.DataLoader`` yielding batches
            of ``pixel_values``, ``input_ids`` and ``attention_mask``.
        model_parts: Tuple of ``(noise_scheduler, text_encoder, vae, unet)``,
            as returned by ``retrieve_model_parts``.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        The fine-tuned ``UNet2DConditionModel``, unwrapped from any
        ``Accelerator`` distributed-training wrapper.
    """
    logger.info("Executing start_fine_tunning")
    noise_scheduler, text_encoder, vae, unet = model_parts

    # __, noise_scheduler, text_encoder, vae, unet = retrieve_model_parts(device)

    accelerator, unet, optimizer = __prepare_training(unet, traindata_loader)

    for epoch in range(training_epochs):
        logger.info("Epoch %d executing", epoch)
        progress_bar = tqdm(traindata_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Se pasan los pixeles al espacio latente con el encoder del VAE:
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

            # Proceso de difusion hacia delante:
            # 1. Creamos ruido aleatorio
            noise = torch.randn_like(latents)
            # 2. Cogemos un timestep aleatorio:
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
            # 3. Añadimos ruido al vector del espacio latente:
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Codificamos el texto:
            input_ids = batch["input_ids"].to(accelerator.device)
            encoder_hidden_states = text_encoder(input_ids)[0]

            # Con el vector con ruido, el timestep, y el vector de texto, hacemos la prediccion de ruido:
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calculamos el error y actualizamos los parametros:
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

    unwrapped_unet = accelerator.unwrap_model(unet)
    logger.info("Finished start_fine_tunning")
    return unwrapped_unet
