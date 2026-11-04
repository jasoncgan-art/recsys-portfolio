import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="CSV with video_id,image_path; paths relative to manifest")
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    weights = ViT_B_16_Weights.DEFAULT
    preprocess = weights.transforms()
    model = vit_b_16(weights=weights).eval().to(args.device)
    model.heads = torch.nn.Identity()
    vectors = {}
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream, torch.inference_mode():
        for row in csv.DictReader(stream):
            with Image.open(args.manifest.parent / row["image_path"]) as image:
                x = preprocess(image.convert("RGB")).unsqueeze(0).to(args.device)
                vectors[int(row["video_id"])] = model(x).squeeze(0).cpu()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, args.out)


if __name__ == "__main__":
    main()
