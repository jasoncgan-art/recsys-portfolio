# Sequential Recommendation with Visual Features

Compare a GRU baseline with a causal self-attention model. An optional visual branch adds frozen ViT-B/16 representations to historical item embeddings. The three models use the same splits and candidate catalog.

```sh
pip install -r requirements.txt
python ../make_demo.py
python train.py data/demo.csv --length 10 --epochs 1
```

For real data, pass an extracted KuaiRand log with `time_ms,user_id,video_id,is_click`. Repeat at lengths 10 and 50 with separate `--out` files. On a large catalog, start with a documented subset because exact full-catalog cross-entropy and evaluation scale with the number of training items. For each positive event, the model sees strictly earlier positive events by the same user. The catalog is fitted on the training period. Report HR@10 on warm-catalog targets with nonempty history, and document excluded events before comparing settings. The baseline is a GRU model, not an exact reproduction of GRU4Rec.

## Visual data

The public KuaiRand logs do not establish access to raw frames. To run the visual branch, supply your own permitted item images with matching IDs. A manifest CSV has columns `video_id,image_path`, with paths relative to the manifest. Record the source and coverage. An alternative dataset must provide timestamped histories and usable images keyed by item ID; adapt its records to the schema and use the actual dataset name in your write-up.

```sh
python embed_images.py data/images.csv artifacts/vit.pt --device cuda
python train.py data/events.csv --images artifacts/vit.pt --length 50 --device cuda
```

The first run downloads pretrained torchvision weights. The experiment writes JSON results for ID-only and visual models. ViT fine-tuning, Qdrant serving and W&B tracking are outside this implementation.
