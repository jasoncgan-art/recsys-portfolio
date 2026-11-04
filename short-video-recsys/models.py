import torch
from torch import nn
from torch.nn import functional as F


class Retriever(nn.Module):
    def __init__(self, users, items, tower=True, dim=32):
        super().__init__()
        self.users = nn.Embedding(users, dim)
        self.items = nn.Embedding(items, dim)
        def encoder():
            return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.user_net = encoder() if tower else nn.Identity()
        self.item_net = encoder() if tower else nn.Identity()

    def user(self, ids):
        return self.user_net(self.users(ids))

    def item(self, ids):
        return self.item_net(self.items(ids))

    def forward(self, users, items):
        return (self.user(users) * self.item(items)).sum(-1)


class Ranker(nn.Module):
    def __init__(self, users, items, cross=True, dim=32):
        super().__init__()
        self.users = nn.Embedding(users, dim)
        self.items = nn.Embedding(items, dim)
        width = 2 * dim
        self.cross = nn.ModuleList(nn.Linear(width, width) for _ in range(2 if cross else 0))
        self.deep = nn.Sequential(nn.Linear(width, width), nn.ReLU())
        self.head = nn.Linear(width * (2 if cross else 1), 2)

    def forward(self, users, items):
        base = torch.cat((self.users(users), self.items(items)), dim=-1)
        crossed = base
        for layer in self.cross:
            crossed = crossed + base * layer(crossed)
        deep = self.deep(base)
        return self.head(torch.cat((deep, crossed), -1) if self.cross else deep)


def ranking_score(logits):
    return logits[..., 0].sigmoid() * F.softplus(logits[..., 1])
