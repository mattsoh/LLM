from flask import Flask, request, render_template_string
import torch
import tiktoken
import gdown
import os
from helper import generate_and_print_sample, GPT

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

MODEL_PATH = "model.pth"
FILE_ID = "1mkhifEI6HQoiVahnGgZaRYsVU23req_M"

if not os.path.exists(MODEL_PATH):
    try:
        print("Downloading model.pth from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, verify=False)
        print("Download completed.")
    except Exception as e:
        print(f"Error downloading model.pth: {e}")
else:
    print("model.pth already exists.")

device = torch.device("cpu")
model = GPT(GPT_CONFIG)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
tokenizer = tiktoken.get_encoding("gpt2")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_context = request.form['start_context']
        result = generate_and_print_sample(model, tokenizer, device, start_context)
        return render_template_string('''
            <form method="post">
                Start Context: <input type="text" name="start_context">
                <input type="submit" value="Generate">
            </form>
            <p>{{ result }}</p>
        ''', result=result)
    return render_template_string('''
        <form method="post">
            Start Context: <input type="text" name="start_context">
            <input type="submit" value="Generate">
        </form>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
