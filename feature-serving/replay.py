import argparse
import csv
import itertools
import json
from collections import defaultdict, deque
from pathlib import Path

WINDOW_MS = 600_000


def replay(rows):
    """Emit counts over [timestamp - WINDOW_MS, timestamp) before applying events."""
    histories = defaultdict(deque)
    last = None
    for timestamp, events in itertools.groupby(rows, key=lambda row: row[0]):
        if last is not None and timestamp < last:
            raise ValueError("Replay input must be sorted by event time")
        last = timestamp
        batch = list(events)
        for _, user in batch:
            history = histories[user]
            while history and history[0] < timestamp - WINDOW_MS:
                history.popleft()
            yield timestamp, user, len(history)
        for _, user in batch:
            histories[user].append(timestamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--redis-url", help="Optional Redis snapshot; use a dedicated demo database")
    args = parser.parse_args()
    redis_client = None
    if args.redis_url:
        import redis
        redis_client = redis.Redis.from_url(args.redis_url, socket_timeout=1)
    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        rows = sorted((int(row["time_ms"]), int(row["user_id"])) for row in csv.DictReader(stream))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {}
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_ms", "user_id", "prior_events_10m"])
        for t, user, count in replay(rows):
            writer.writerow([t, user, count])
            snapshot[user] = json.dumps({"as_of_ms": t, "prior_events_10m": count})
    if redis_client and snapshot:
        redis_client.mset({f"recsys:replay:user:{user}": payload for user, payload in snapshot.items()})
    print(f"Wrote {len(rows)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
