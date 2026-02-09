import torch
import pickle
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from peft import PeftModel

MODEL_DIR = "tier_c_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# ---------------- LOAD BASE MODEL ----------------
base_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)

# ---------------- LOAD LORA ADAPTER ----------------
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

model.to(device)
model.eval()

# ---------------- LOAD LABEL ENCODER ----------------
with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
