import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SequenceRanker(nn.Module):
    def __init__(self, count, length, architecture, images=None, dim=64):
        super().__init__()
        self.emb = nn.Embedding(count, dim, padding_idx=0)
        self.position = nn.Embedding(length, dim)
        self.architecture = architecture
        self.encoder = (nn.GRU(dim, dim, batch_first=True) if architecture == "gru"
                        else nn.TransformerEncoder(nn.TransformerEncoderLayer(
                            dim, 4, 2 * dim, batch_first=True, dropout=.1), 2,
                            enable_nested_tensor=False))
        self.register_buffer("images", images)
        self.image_projection = nn.Linear(images.shape[-1], dim, bias=False) if images is not None else None

    def forward(self, history):
        length = history.shape[1]
        x = self.emb(history)
        if self.image_projection is not None:
            x = x + self.image_projection(self.images[history])
        if self.architecture == "gru":
            x, _ = self.encoder(x)
        else:
            x = x + self.position(torch.arange(length, device=x.device))
            mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            x = self.encoder(x, mask=mask, src_key_padding_mask=history.eq(0))
        end = history.ne(0).sum(1) - 1
        return x[torch.arange(len(history), device=x.device), end] @ self.emb.weight.T


def load(path, length):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = sorted((int(r["time_ms"]), int(r["user_id"]), int(r["video_id"]))
                      for r in csv.DictReader(stream) if int(r["is_click"]))
    if len(rows) < 10:
        raise ValueError("Expected at least ten positive events")
    cutoff = [rows[int(len(rows) * q)][0] for q in (.8, .9)]
    items = {i: n + 1 for n, i in enumerate(sorted({i for t, _, i in rows if t < cutoff[0]}))}
    history = defaultdict(list)
    partitions = [([], []) for _ in range(3)]
    from itertools import groupby
    for t, events in groupby(rows, key=lambda row: row[0]):
        pending = []
        for _, u, i in events:
            if i not in items:
                continue
            context = history[u][-length:]
            if context:
                x, y = partitions[sum(t >= boundary for boundary in cutoff)]
                x.append(context + [0] * (length - len(context)))
                y.append(items[i])
            pending.append((u, items[i]))
        for u, i in pending:
            history[u].append(i)
    if any(not x for x, _ in partitions):
        raise ValueError("Every temporal partition needs known targets with nonempty history")
    return items, [TensorDataset(torch.tensor(x), torch.tensor(y)) for x, y in partitions]


def run(model, partitions, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in DataLoader(partitions[0], batch_size=128, shuffle=True):
            loss = loss_fn(model(x.to(device))[:, 1:], y.to(device) - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    results = {}
    with torch.inference_mode():
        for name, data in zip(("validation", "test"), partitions[1:]):
            hits = 0
            for x, y in DataLoader(data, batch_size=128):
                scores = model(x.to(device))[:, 1:]
                top = scores.topk(min(10, scores.shape[1])).indices.cpu() + 1
                hits += int(top.eq(y[:, None]).any(1).sum())
            results[name] = {"examples": len(data), "hits": hits, "hr_at_10": hits / len(data)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--images", help="Optional trusted tensor bundle from embed_images.py")
    parser.add_argument("--out", type=Path, default=Path("artifacts/metrics.json"))
    args = parser.parse_args()
    if args.length < 1:
        parser.error("--length must be positive")
    items, partitions = load(args.csv, args.length)
    images = None
    coverage = 0
    if args.images:
        source = torch.load(args.images, weights_only=True)
        images = torch.zeros(len(items) + 1, 768)
        for raw_id, vector in source.items():
            if raw_id in items:
                images[items[raw_id]] = vector
                coverage += 1
        if not coverage:
            parser.error("No image IDs match the training catalog")
    report = {"history_length": args.length, "catalog": len(items), "items_with_images": coverage}
    for label, arch, features in (("gru", "gru", None), ("sasrec", "sasrec", None),
                                  ("sasrec_visual", "sasrec", images)):
        if label == "sasrec_visual" and features is None:
            continue
        torch.manual_seed(42)
        model = SequenceRanker(len(items) + 1, args.length, arch, features).to(args.device)
        report[label] = run(model, partitions, args.epochs, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
