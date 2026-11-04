# Recommender Systems Portfolio

Three related student projects. The short-video system trains a model artifact that the feature-serving project can benchmark. The sequential project runs its own chronological next-item experiment.

| Project | Runnable entry points |
| --- | --- |
| [Short-video recommendation](short-video-recsys/) | Train and evaluate retrieval and ranking; serve HNSW results through FastAPI |
| [Sequential recommendation](sequential-recsys/) | Compare GRU, ID-only SASRec and SASRec with optional visual features |
| [Feature serving](feature-serving/) | Replay point-in-time features, export Redis snapshots and benchmark ONNX FP32 versus FP16 |

Run `python make_demo.py` here to create a deterministic synthetic fixture in `data/demo.csv`. It checks mechanics and has no bearing on the resume benchmarks. Read each project README for its data, protocol, commands and remaining experiment work.
