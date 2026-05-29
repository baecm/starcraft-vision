# A Saliency-Prior Auxiliary Objective for Esports Observing

Esports spectating in real-time strategy (RTS) games requires selecting a fixed-size camera view under high concurrency and tight time constraints.
We formulate automatic observing as viewport prediction over a tile-aligned, multi-channel game-state tensor supervised by aggregated human viewports.
Building on detector-style behavior cloning with Mask R-CNN, we propose Kernel-Based Region Scoring (KBRS), a lightweight auxiliary loss-requiring only fixed-kernel operations without additional learned parameters-that injects domain structure during training while leaving inference-time behavior unchanged.
KBRS computes an explicit score field on backbone feature maps using these fixed-kernel operations, which encode three visual cues: local activity density, opponent-ally mixture, and centeredness.
This score field is used only during training via a small auxiliary loss to shape intermediate representations, leaving the detection architecture and inference-time outputs unchanged.
Experiments on StarCraft: Brood War replays processed into synchronized state tensors and human viewports show that KBRS improves agreement with aggregated human attention.
Averaged over three replay-level folds and three random seeds, KBRS yields consistent gains in tile-level viewport overlap (Intersection Ratio and Intersection@any, 0.30, 0.50) over the Mask R-CNN imitation baseline, with only modest additional training cost.

## Usage
### Training and Inference
This command trains the configured model and subsequently runs inference using the resulting model.
```bash
# Train and Inference with reproducible settings
bash ./entrypoint.sh run \
  -m \
  dataset=fold1_sample \
  model=kbrs \
  mode=train_and_inference \
  seed=123 \
  kbrs_loss=kbrs025 \
  kbrs_score=base \
  kbrs_score.density=0.3 \
  kbrs_score.mixture=3.0 \
  kbrs_score.centeredness=0.3 \
  max_epoch=1 \
  num_workers=0 \
  +score_threshold=0.0
```
[parameters]

Detail configurations are able to find at `code/conf/`

- `--mode`: ["train_only", "inference_only", 'train_and_inference"]
- `--dataset`: set train and test dataset.
- `--model`: ["vanilla", "kbrs"]
- `--seed`: set the global run seed

Below parameters are used for training saliency-prior(Kernel-based Region Scoring) model
- `--kbrs_loss`: 
- `--kbrs_score`:
- `--kbrs_score.density`:
- `--kbrs_score.mixture`:
- `--kbrs_score.centeredness`:

`--max_epoch`, `--num_workers`, `score_threshhold` are temporary configuration for small-scale run.


### Estimate
After training and inference, estimate the performance of the predicted results.

```bash
bash ./entrypoint.sh estimate \
  --mode model \
  --model-name {MODEL_NAME} \
  --epoch {EPOCH} \
  --replays {REPLAYS} \
  --input-root /data/input/dst \
  --label-root /data/label/dst \
  --pred-root /results/predictions \
  --csv-out /results/final_metrics.csv
```

[parameter]
- `--model`: 
- `--model-name`: 
- `--replays`: List,
- `--input-root`: 
- `--label-root`: 
- `--pred-root`: 
- `--csv-out`: 

## Dataset