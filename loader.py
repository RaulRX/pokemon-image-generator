# Cargando el dataset de Pokemon con texto:
dataset_name = "reach-vb/pokemon-blip-captions"
max_train_samples = 200  # tiene que ser menor que 833 (el numero de filas que tiene el dataset original)


dataset = load_dataset(dataset_name, split="train")
if max_train_samples:
    dataset = dataset.select(range())

# Comprobamos el tamaño de las imagenes del dataset:
size = dataset[0]["image"].size
print(f"Tamaño de las imagenes del dataset: {size}")

# Creamos un Dataset wrapper para la hora del entrenamiento:
batch_size = 2  # no puede ser mayor por la limitacion de GPU

class Text2ImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = image_transforms(example["image"].convert("RGB"))
        token = tokenizer(example["text"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        return {
            "pixel_values": image,
            "input_ids": token.input_ids.squeeze(0),
            "attention_mask": token.attention_mask.squeeze(0),
        }

train_dataset = Text2ImageDataset(dataset)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Tokenizador:
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder='tokenizer')

# Scheduler:
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

# Text Encoder (CLIP):
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name,
    subfolder="text_encoder",
).to(device)

# VAE: Autoencoder:
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name,
    subfolder="vae",
).to(device)

# La UNet:
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name,
    subfolder="unet",
).to(device)

# Congelamos los pesos del VAE y del Text Encoder, ya que solo queremos finetunear la UNet:
vae.eval()
text_encoder.eval()

for param in vae.parameters():
    param.requires_grad = False
for param in text_encoder.parameters():
    param.requires_grad = False