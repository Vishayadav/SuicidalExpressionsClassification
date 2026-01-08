import argparse
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


def build_dataloaders(processed_path: Path, batch_size: int, val_size: float, max_train_samples: int = None, random_state: int = 42):
    data = torch.load(processed_path)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    labels = data['labels']

    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    labels = labels.cpu()

    idx = list(range(len(labels)))
    train_idx, val_idx = train_test_split(idx, test_size=val_size, stratify=labels.numpy(), random_state=random_state)

    if max_train_samples is not None and max_train_samples > 0:
        train_idx = train_idx[:max_train_samples]

    train_ds = TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx])
    val_ds = TensorDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    losses = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for input_ids, attn, labels in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            losses += loss.item() * labels.size(0)
    return losses / total, correct / total


def train(args):
    device = torch.device('cpu')
    print('Using device:', device)

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer)
    model = DistilBertForSequenceClassification.from_pretrained(args.tokenizer, num_labels=2)
    model.to(device)

    train_loader, val_loader = build_dataloaders(Path(args.processed), batch_size=args.batch_size, val_size=args.val_size, max_train_samples=args.max_train_samples)

    optim = AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        running_loss = 0.0
        for input_ids, attn, labels in pbar:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f'End epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        # save every epoch
        model.save_pretrained(outdir)
        tokenizer.save_pretrained(outdir)

        if val_acc > best_acc:
            best_acc = val_acc
            print(f'New best acc {best_acc:.4f}, model saved to {outdir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=str, default='processed_test.pt')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-uncased')
    parser.add_argument('--output_dir', type=str, default='model')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--max_train_samples', type=int, default=2000, help='limit number of training samples for speed on CPU (use 0 or negative for no limit)')
    args = parser.parse_args()

    if args.max_train_samples <= 0:
        args.max_train_samples = None

    train(args)


if __name__ == '__main__':
    main()
