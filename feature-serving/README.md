# Point-in-Time Features and Inference Benchmark

Replay timestamped events to build a rolling ten-minute user feature. Before applying any events at time t, the replay emits counts in [t - 10 minutes, t) for each event. This keeps current and same-timestamp events out of the feature. Optionally materialize the last emitted value per user in a demo Redis database.

```sh
python ../make_demo.py
python replay.py data/demo.csv artifacts/features.csv
python -m unittest -v
```

Pass `--redis-url redis://localhost:6379/0` to write the snapshot. This is a sorted offline replay. A continuously updated service would additionally need watermarks, a late-event policy, a feature expiration policy, and tests for offline/online parity. Feast, PySpark, DuckDB and Triton are potential extensions. The current path uses Python and Redis directly.

## GPU inference experiment

Train the ranker in the short-video project first. Use `pip install -r requirements-gpu.txt` in a separate environment and select an ONNX Runtime GPU build compatible with your CUDA/cuDNN environment.

```sh
python bench_onnx.py ../short-video-recsys/artifacts/models.pt --model-dir ../short-video-recsys --batch 128
```

The script exports fixed-shape FP32 and FP16 variants with identical inputs, warms up each session, and records p95 batch latency, item throughput, file sizes and a maximum logit delta. Host-device transfers are included. File size is distinct from peak GPU memory. Before claiming VRAM or NDCG changes, profile peak allocation and score held-out candidates with both models. The microbenchmark uses synthetic ID pairs and does not measure end-to-end service latency.
