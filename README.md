# Diffusion Transformer (DiT) with Magnitude-Preserving Layers

 - [Magnitude preservation paper](https://arxiv.org/abs/2312.02696)
 - [DiT paper](https://arxiv.org/abs/2212.09748)

## Training

```bash
python train.py --data-path /path/to/data --results-dir /path/to/results --model DiT-XS/2
```

## Sampling

```bash
python sample.py --result-dir /path/to/result --ckpt <checkpoint>.pt --class-label 2 --seed 42 --cfg-scale 1.5
```
