import argparse
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def prepare_loaders(processed_path: Path, batch_size: int = 16, val_size: float = 0.1, random_state: int = 42):
    data = torch.load(processed_path)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    labels = data['labels']

    # convert to CPU tensors if needed
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    labels = labels.cpu()

    idx = list(range(len(labels)))
    train_idx, val_idx = train_test_split(idx, test_size=val_size, stratify=labels.numpy(), random_state=random_state)

    def subset(indices):
        return TensorDataset(input_ids[indices], attention_mask[indices], labels[indices])

    train_ds = subset(train_idx)
    val_ds = subset(val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=str, default='processed_test.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    train_loader, val_loader = prepare_loaders(Path(args.processed), batch_size=args.batch_size)
    print('Train batches:', len(train_loader), 'Val batches:', len(val_loader))
    for b in train_loader:
        input_ids, attn, labels = b
        print('batch shapes -> input_ids, attention_mask, labels:', input_ids.shape, attn.shape, labels.shape)
        break


if __name__ == '__main__':
    main()
