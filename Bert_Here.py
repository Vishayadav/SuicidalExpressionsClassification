import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# --------------------------------------------------
# 1. DATASET
# --------------------------------------------------

texts = [
    "I feel like giving up on everything",
    "Nothing matters anymore",
    "I want to disappear",
    "I feel empty and hopeless",
    "Life is pointless",
    "I am feeling good today",
    "I love spending time with my friends",
    "I am excited about my future",
    "Today was productive",
    "I enjoy learning new things"
]

# 1 = suicidal, 0 = non-suicidal
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# --------------------------------------------------
# 2. TOKENIZER
# --------------------------------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

# --------------------------------------------------
# 3. DATASET CLASS
# --------------------------------------------------

class SuicideDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

dataset = SuicideDataset(encodings, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# --------------------------------------------------
# 4. MODEL
# --------------------------------------------------

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

device = torch.device("cpu")
model.to(device)

# --------------------------------------------------
# 5. OPTIMIZER
# --------------------------------------------------

optimizer = AdamW(model.parameters(), lr=5e-5)

# --------------------------------------------------
# 6. TRAINING LOOP (PURE PYTORCH)
# --------------------------------------------------

epochs = 3

print("\nTraining BERT...\n")

model.train()
for epoch in range(epochs):
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

print("\nTraining complete.\n")

# --------------------------------------------------
# 7. PREDICTION FUNCTION
# --------------------------------------------------

def predict(text):
    model.eval()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )

    probs = torch.softmax(outputs.logits, dim=1)

    return {
        "non_suicidal_probability": round(probs[0][0].item(), 4),
        "suicidal_probability": round(probs[0][1].item(), 4)
    }

# --------------------------------------------------
# 8. INTERACTIVE LOOP
# --------------------------------------------------

while True:
    text = input("\nEnter text (or type 'exit'): ")
    if text.lower() == "exit":
        break

    print(predict(text))
