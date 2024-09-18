import os
import torch
import urllib.request
import tiktoken
from helper import create_dataloader_v1, create_dataloader, train_model_simple, GPTModel

GPT_CONFIG = {
    "vocab_size": 50257,    # Weird but I forgot to change it
    "context_length": 256,  # Apparently it's shortened but idk
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,  # VERY VERY VERY VERY IMPORTANT!!!!!!!!
    "batch_size": 2,
    "weight_decay": 0.1
}

TRAIN_RATIO = 0.90

# DON'T CHANGE FROM HERE

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRUST ME, NOT YOUR FILES 

file_path = "world.txt"
url = "https://raw.githubusercontent.com/mattsoh/LLM/main/world.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

model = GPTModel(GPT_CONFIG)
model.to(device)
if os.path.exists("model.pth"):
    print("Loading")
    model.load_state_dict(torch.load("model.pth", map_location=device))
else:
    print("Creating new model")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
)

split_idx = int(TRAIN_RATIO * len(text_data))

train_loader = create_dataloader_v1(
    text_data[:split_idx],
    batch_size=SETTINGS["batch_size"],
    max_length=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    text_data[split_idx:],
    batch_size=SETTINGS["batch_size"],
    max_length=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

tokenizer = tiktoken.get_encoding("gpt2")

train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=SETTINGS["num_epochs"], eval_freq=5, eval_iter=1,
    start_context="To be", tokenizer=tokenizer
)


# SAVE SAVE SAVE SAVE SAVE

torch.save(model.state_dict(), "model.pth")
