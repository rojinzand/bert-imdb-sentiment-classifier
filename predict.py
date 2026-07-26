"""
predict.py

Loads the fine-tuned model from ./best_model/ and predicts the sentiment
of a given piece of text.

Usage:
    python predict.py
    (edit the `text` variable below, or adapt to take input() / sys.argv)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BEST_MODEL_PATH = "best_model"
MAX_LENGTH = 128

LABELS = ["negative", "positive"]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    label = LABELS[predicted_class.item()]
    return label, confidence.item()


if __name__ == "__main__":
    tokenizer, model, device = load_model()

    text = "I absolutely loved this movie."

    label, confidence = predict(text, tokenizer, model, device)
    print(f"Input: {text}")
    print(f"Prediction: {label.capitalize()} ({confidence*100:.1f}%)")
