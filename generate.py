from flask import Flask, request, render_template_string
from torch import device, load
from tiktoken import get_encoding
from gdown import download
import os
from helper_simple import generate_and_print_sample, GPT

app = Flask(__name__)

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

FILE_ID = "1mkhifEI6HQoiVahnGgZaRYsVU23req_M"
MODEL_PATH = "model.pth"
if not os.path.exists(MODEL_PATH):
    try:
        print("Downloading model.pth from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        download(url, MODEL_PATH, quiet=False, verify=False)
        print("Download completed.")
    except Exception as e:
        print(f"Error downloading model.pth: {e}")
else:
    print("model.pth already exists.")
device = torch.device("cpu")
model = GPT(GPT_CONFIG)
model.to(device)
model.load_state_dict(load(MODEL_PATH, map_location=device, weights_only=True))
tokenizer = get_encoding("gpt2")

start_context = input("Start Context: ")
result = generate_and_print_sample(model, tokenizer, device, start_context)
print(result)