"""Generate a small deterministic fixture for pipeline smoke checks."""
import csv
import random
from pathlib import Path

path = Path("data/demo.csv")
path.parent.mkdir(exist_ok=True)
rng = random.Random(42)
with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["time_ms", "user_id", "video_id", "is_click", "play_time_ms"])
    for n in range(5000):
        user, item = rng.randrange(24), rng.randrange(140)
        click = int(rng.random() < (.72 if user % 7 == item % 7 else .21))
        writer.writerow([1000 * n, user, item, click, rng.randrange(500, 20000) if click else 0])
print(path)
