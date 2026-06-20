
import os
import torch
from accelerate import Accelerator
from tqdm import tqdm
from src.loader import retrieve_model_parts

learning_rate = float(os.getenv("LEARNING_RATE", "1e-5")) 
training_epochs = int(os.getenv("TRAIN_EPOCHS", "2"))

def __prepare_training(unet, traindata_loader):
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # Acelerador:
    accelerator = Accelerator()
    unet, optimizer, __ = accelerator.prepare(unet, optimizer, traindata_loader)
    print(accelerator.device)

    return accelerator, unet, optimizer

def start_fine_tunning(traindata_loader, device="cpu"):

    __, noise_scheduler, text_encoder, vae, unet = retrieve_model_parts(device)

    accelerator, unet, optimizer = __prepare_training(unet, traindata_loader)

    for epoch in range(training_epochs):
        progress_bar = tqdm(traindata_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:

            # Se pasan los pixeles al espacio latente con el encoder del VAE:
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                latents = latents * 0.18215

            # Proceso de difusion hacia delante:
            # 1. Creamos ruido aleatorio
            noise = torch.randn_like(latents)
            # 2. Cogemos un timestep aleatorio:
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            # 3. Añadimos ruido al vector del espacio latente:
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Codificamos el texto:
            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

            # Con el vector con ruido, el timestep, y el vector de texto, hacemos la prediccion de ruido:
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calculamos el error y actualizamos los parametros:
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

    return accelerator.unwrap_model(unet)