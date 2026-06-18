# Optimizador:
learning_rate = 1e-5
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# Acelerador:
accelerator = Accelerator()
unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
print(accelerator.device)