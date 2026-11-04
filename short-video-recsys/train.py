import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import Ranker, Retriever, ranking_score


def load(paths, limit):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                if limit and index >= limit:
                    break
                rows.append((int(row["time_ms"]), int(row["user_id"]), int(row["video_id"]),
                             int(row["is_click"]), max(0, int(row["play_time_ms"]))))
    return sorted(rows)


def fit(model, data, epochs, device, ranker=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for _ in range(epochs):
        for batch in DataLoader(data, batch_size=1024, shuffle=True):
            users, items, clicks, watch = (x.to(device) for x in batch)
            pred = model(users, items)
            if ranker:
                loss = (F.binary_cross_entropy_with_logits(pred[:, 0], clicks)
                        + F.smooth_l1_loss(F.softplus(pred[:, 1]), watch))
            else:
                loss = F.binary_cross_entropy_with_logits(pred, clicks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.inference_mode()
def evaluate(retrievers, rankers, rows, catalog_size, device):
    positives = {}
    for _, u, i, click, _ in rows:
        if click:
            positives.setdefault(u, set()).add(i)
    if not positives:
        raise ValueError("Evaluation needs positive engagement on known users and items")
    scores = {}
    for name, model in retrievers.items():
        catalog = model.item(torch.arange(catalog_size, device=device))
        total = 0.
        ndcg = {name: 0. for name in rankers}
        for u, relevant in positives.items():
            user = torch.tensor([u], device=device)
            ids = (model.user(user) @ catalog.T).flatten().topk(min(100, catalog_size)).indices
            total += len(set(ids.tolist()) & relevant) / len(relevant)
            if name != "two_tower":
                continue
            ideal = sum(1 / math.log2(j + 2) for j in range(min(10, len(relevant))))
            for label, ranker in rankers.items():
                order = ranking_score(ranker(user.expand_as(ids), ids)).argsort(descending=True)
                ndcg[label] += sum((item in relevant) / math.log2(j + 2)
                                   for j, item in enumerate(ids[order[:10]].tolist())) / ideal
        scores[name + "_recall_at_100"] = total / len(positives)
        if name == "two_tower":
            scores.update({label + "_ndcg_at_10": value / len(positives)
                           for label, value in ndcg.items()})
    return {"positive_users": len(positives), **scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--limit-per-file", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = load(args.csv, args.limit_per_file)
    if len(rows) < 10:
        parser.error("At least 10 rows are required")
    val_time, test_time = [rows[int(len(rows) * p)][0] for p in (.8, .9)]
    train = [row for row in rows if row[0] < val_time]
    if not train or val_time == test_time:
        parser.error("Timestamp ties leave a temporal partition empty")
    users = {k: i for i, k in enumerate(sorted({r[1] for r in train}))}
    items = {k: i for i, k in enumerate(sorted({r[2] for r in train}))}
    def encode(part):
        return [(t, users[u], items[i], c, w) for t, u, i, c, w in part
                if u in users and i in items]
    _, u, i, c, w = map(torch.tensor, zip(*encode(train)))
    data = TensorDataset(u, i, c.float(), (w.float() / 1000).log1p())
    retrievers = {name: Retriever(len(users), len(items), tower).to(args.device)
                  for name, tower in (("mf", False), ("two_tower", True))}
    rankers = {name: Ranker(len(users), len(items), cross).to(args.device)
               for name, cross in (("mlp", False), ("dcnv2", True))}
    for model in retrievers.values():
        fit(model, data, args.epochs, args.device)
    for model in rankers.values():
        fit(model, data, args.epochs, args.device, ranker=True)
    val = [r for r in rows if val_time <= r[0] < test_time]
    test = [r for r in rows if r[0] >= test_time]
    report = {"seed": 42, "epochs": args.epochs, "train_rows": len(train),
              "catalog_size": len(items), "validation_rows": len(val), "test_rows": len(test),
              "test_known_rows": len(encode(test)), "validation_start_ms": val_time,
              "test_start_ms": test_time,
              "validation": evaluate(retrievers, rankers, encode(val), len(items), args.device),
              "test": evaluate(retrievers, rankers, encode(test), len(items), args.device)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    torch.save({"users": users, "items": items, "tower": retrievers["two_tower"].cpu().state_dict(),
                "ranker": rankers["dcnv2"].cpu().state_dict()}, args.out / "models.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
