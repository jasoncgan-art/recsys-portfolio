import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen


def request(url, scheduled_at):
    started = time.perf_counter()
    try:
        with urlopen(url, timeout=5) as response:
            response.read()
        return time.perf_counter() - started, started - scheduled_at
    except OSError:
        return None, started - scheduled_at


def p95(values):
    return round(1000 * sorted(values)[math.ceil(.95 * len(values)) - 1], 3) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/recommend/0")
    parser.add_argument("--qps", type=int, default=50)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if min(args.qps, args.seconds, args.workers) < 1:
        parser.error("Use positive load parameters")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        start = time.perf_counter()
        futures = []
        for i in range(args.qps * args.seconds):
            target = start + i / args.qps
            time.sleep(max(0, target - time.perf_counter()))
            futures.append(pool.submit(request, args.url, target))
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start
    success = [latency for latency, _ in results if latency is not None]
    print(json.dumps({"target_requests_per_second": args.qps, "requests": len(results),
                      "workers": args.workers, "actual_successes_per_second": len(success) / elapsed,
                      "errors": len(results) - len(success), "elapsed_seconds": elapsed,
                      "p95_http_ms": p95(success),
                      "p95_client_queue_ms": p95([delay for _, delay in results])}, indent=2))


if __name__ == "__main__":
    main()
