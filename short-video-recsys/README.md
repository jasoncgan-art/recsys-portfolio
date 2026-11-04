# Two-Stage Short-Video Recommendation

A PyTorch dual-encoder retrieves candidates and a multitask ranker combines predicted engagement with predicted log watch seconds. Matrix factorization and an MLP provide baselines. The offline evaluation uses exact retrieval, while the API uses FAISS HNSW.

```sh
pip install -r requirements.txt
python ../make_demo.py
python train.py data/demo.csv --epochs 1
python -m uvicorn serve:app --port 8000
```

From another shell, send a GET request to `http://localhost:8000/recommend/0`. The service accepts original dataset user IDs. For a container, build with `docker build -t short-video-recsys .` and mount the training output at `/app/artifacts`. Optional `REDIS_URL` enables a user-vector cache. Give every deployed checkpoint a distinct `CACHE_NAMESPACE`.

For a local single-user HTTP smoke benchmark, warm the endpoint and run `python load_test.py --qps 50 --seconds 30`. This client opens a connection per request. Inspect client queue time and achieved throughput to see whether it kept up.

## Public data

Download [KuaiRand-1K](https://kuairand.com/) under its terms. Supply one or more extracted logs as positional arguments, for example `python train.py data/log_standard_4_08_to_4_21.csv data/log_standard_4_22_to_5_08.csv --device cuda`. Start with a manageable subset; the loader holds rows in memory and exact retrieval scores the training catalog. `--limit-per-file` takes the first N rows per file, which is useful for development and is not a random sample.

## Protocol

Global chronological 80/10/10 boundaries keep equal timestamps together. Training determines known users and items. The report includes test-row coverage after removing unknown IDs. Both retrieval models learn from observed impressions. In KuaiRand, the `is_click` field can mean a valid play depending on the interface, so treat it as an engagement flag unless you explicitly filter the UI. Rankers see identical training labels and rerank the same two-tower candidates. Relevance is binary observed engagement per user in the held-out window. Unobserved items have unknown relevance and receive no credit in this offline metric. Users without known positive held-out items are excluded.

Before reporting a production-style result, define candidates and relevance per request, compare HNSW with exact retrieval, run repeated seeds, and record latency at a stated request rate, duration, hardware, catalog size and error rate. This code reports offline scores and exposes an endpoint. The resume's measured improvements and latency claims require separate runs.
