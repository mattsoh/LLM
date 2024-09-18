import torch
import tiktoken
from helper import generate_and_print_sample, text_to_token_ids, token_ids_to_text, generate, GPT

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

device = torch.device("cpu")
model = GPT(GPT_CONFIG)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
START_CONTEXT = "Every day is a new day"
tokenizer = tiktoken.get_encoding("gpt2")
generate_and_print_sample(model, tokenizer, device, START_CONTEXT)
