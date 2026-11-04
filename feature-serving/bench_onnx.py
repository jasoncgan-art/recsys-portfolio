import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if min(args.batch, args.runs) <= 0:
        parser.error("Batch and run count must be positive")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        parser.error("A working ONNX Runtime CUDA provider is required")
    sys.path.insert(0, str(args.model_dir.resolve()))
    Ranker = importlib.import_module("models").Ranker
    bundle = torch.load(args.checkpoint, weights_only=True)
    ranker = Ranker(len(bundle["users"]), len(bundle["items"])).eval()
    ranker.load_state_dict(bundle["ranker"])
    args.out.mkdir(parents=True, exist_ok=True)
    original = args.out / "ranker_fp32.onnx"
    converted = args.out / "ranker_fp16.onnx"
    ids = torch.zeros(args.batch, dtype=torch.long)
    torch.onnx.export(ranker, (ids, ids), str(original), input_names=["users", "items"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    onnx.save(float16.convert_float_to_float16(onnx.load(original), keep_io_types=True), converted)
    rng = np.random.default_rng(42)
    inputs = {"users": rng.integers(0, len(bundle["users"]), args.batch, dtype=np.int64),
              "items": rng.integers(0, len(bundle["items"]), args.batch, dtype=np.int64)}
    report = {"batch_size": args.batch, "runs": args.runs, "provider": "CUDAExecutionProvider"}
    outputs = []
    for precision, path in (("fp32", original), ("fp16", converted)):
        session = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("CUDA provider did not initialize")
        for _ in range(10):
            output, = session.run(None, inputs)
        outputs.append(output)
        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            session.run(None, inputs)
            timings.append(time.perf_counter() - start)
        report[precision] = {"file_bytes": path.stat().st_size,
                             "p95_ms": float(np.percentile(timings, 95) * 1000),
                             "items_per_second": args.batch * len(timings) / sum(timings)}
        del session
    report["max_abs_logit_delta"] = float(np.abs(outputs[0] - outputs[1]).max())
    (args.out / "onnx_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
