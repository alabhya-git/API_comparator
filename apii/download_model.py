from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

# 🔹 Your Hugging Face token
HF_TOKEN = ""

# 🔹 Model ID
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# 🔹 Authenticate programmatically
login(token=HF_TOKEN)

# Optional: Faster download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    print("Downloading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)

    print("Downloading model (this may take a while)…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        use_auth_token=HF_TOKEN,
        trust_remote_code=True,  # Required for some models
        resume_download=True
    )

    print("✅ Model downloaded successfully!")

except Exception as e:
    print("❌ Download failed:", e)

