import torch
import tiktoken
import gdown
import os
from helper import generate_and_print_sample, GPT

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

MODEL_PATH = "model.pth"


FILE_ID = "1mkhifEI6HQoiVahnGgZaRYsVU23req_M"
MODEL_PATH = "model.pth"

if not os.path.exists(MODEL_PATH):
    try:
        print("Downloading model.pth from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, verify=False)
        print("Download completed.")
    except Exception as e:
        print(f"Error downloading model.pth: {e}")
    
device = torch.device("cpu")
model = GPT(GPT_CONFIG)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
START_CONTEXT = "Every day is a new day"
tokenizer = tiktoken.get_encoding("gpt2")
generate_and_print_sample(model, tokenizer, device, START_CONTEXT)
