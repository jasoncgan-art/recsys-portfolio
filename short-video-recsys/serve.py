import os

import faiss
import redis
import torch
from fastapi import FastAPI, HTTPException

from models import Ranker, Retriever, ranking_score

bundle = torch.load(os.environ.get("MODEL_PATH", "artifacts/models.pt"), weights_only=True)
users, items = bundle["users"], bundle["items"]
item_ids = {index: item for item, index in items.items()}
tower = Retriever(len(users), len(items)).eval()
tower.load_state_dict(bundle["tower"])
ranker = Ranker(len(users), len(items)).eval()
ranker.load_state_dict(bundle["ranker"])
with torch.inference_mode():
    vectors = tower.item(torch.arange(len(items))).numpy()
index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efSearch = 128
index.add(vectors)
cache = redis.Redis.from_url(os.environ["REDIS_URL"], socket_timeout=.25) if "REDIS_URL" in os.environ else None
app = FastAPI()


@app.get("/recommend/{user_id}")
@torch.inference_mode()
def recommend(user_id: int):
    if user_id not in users:
        raise HTTPException(404, "Unknown user")
    user = torch.tensor([users[user_id]])
    vector = None
    key = f"{os.environ.get('CACHE_NAMESPACE', 'dev')}:user:{user_id}"
    if cache:
        try:
            import numpy as np
            raw = cache.get(key)
            if raw and len(raw) == vectors.shape[1] * 4:
                vector = np.frombuffer(raw, dtype=np.float32).reshape(1, -1)
            if vector is None:
                vector = tower.user(user).numpy()
                cache.setex(key, 300, vector.tobytes())
        except redis.RedisError:
            pass
    if vector is None:
        vector = tower.user(user).numpy()
    _, ids = index.search(vector, min(100, len(items)))
    candidates = torch.tensor(ids[0][ids[0] >= 0], dtype=torch.long)
    scores = ranking_score(ranker(user.expand_as(candidates), candidates))
    return {"items": [item_ids[int(i)] for i in candidates[scores.argsort(descending=True)[:10]]]}
