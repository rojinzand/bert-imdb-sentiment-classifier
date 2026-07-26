"""
train.py

Fine-tunes bert-base-uncased on the IMDb dataset for binary sentiment
classification. Saves the best checkpoint (based on validation accuracy)
to ./best_model/, then evaluates it on the held-out test set.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report

# -------------------------------------------------
# Config
# -------------------------------------------------

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
EPOCHS = 5
TRAIN_POOL_SIZE = 8000   # split into train + val below
VAL_SIZE = 1000
TEST_SIZE = 1000
BEST_MODEL_PATH = "best_model"
RANDOM_SEED = 42

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------

dataset = load_dataset("stanfordnlp/imdb")

# -------------------------------------------------
# 2. Load tokenizer
# -------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# -------------------------------------------------
# 3. Build train / val / test subsets
#    Train pool is shuffled and split into train + val.
#    Test set is held out completely and only used at the end.
# -------------------------------------------------

train_pool_raw = dataset["train"].shuffle(seed=RANDOM_SEED).select(range(TRAIN_POOL_SIZE))
test_raw = dataset["test"].shuffle(seed=RANDOM_SEED).select(range(TEST_SIZE))

split = train_pool_raw.train_test_split(test_size=VAL_SIZE, seed=RANDOM_SEED)
train_raw = split["train"]
val_raw = split["test"]

# -------------------------------------------------
# 4. Tokenize each subset (only what we actually use, to save RAM)
# -------------------------------------------------

train_data = train_raw.map(tokenize, batched=True)
val_data = val_raw.map(tokenize, batched=True)
test_data = test_raw.map(tokenize, batched=True)

train_data = train_data.remove_columns(["text"]).rename_column("label", "labels")
val_data = val_data.remove_columns(["text"]).rename_column("label", "labels")
test_data = test_data.remove_columns(["text"]).rename_column("label", "labels")

train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -------------------------------------------------
# 5. DataLoaders
# -------------------------------------------------

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------------------------
# 6. Model + device
# -------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# -------------------------------------------------
# 7. Training + validation loop, saving the best checkpoint
# -------------------------------------------------

best_val_accuracy = 0
best_epoch = 0

for epoch in range(EPOCHS):

    # ---- training ----
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)

    # ---- validation ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    # ---- save only if this epoch beats the previous best ----
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        model.save_pretrained(BEST_MODEL_PATH)
        tokenizer.save_pretrained(BEST_MODEL_PATH)
        print(f"  -> New best model saved (val accuracy: {val_accuracy:.4f})")

print(f"\nBest epoch was {best_epoch} with validation accuracy {best_val_accuracy:.4f}")

# -------------------------------------------------
# 8. Reload the best checkpoint before final testing
# -------------------------------------------------

print(f"\nReloading best model (epoch {best_epoch}) for final test evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_PATH)
model.to(device)

# -------------------------------------------------
# 9. Final evaluation on the held-out test set
# -------------------------------------------------

model.eval()

correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
print(f"\nFinal Test Accuracy: {accuracy:.4f} correct: {correct} total: {total}")
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=["negative", "positive"]))
