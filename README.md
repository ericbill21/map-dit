# MaP-DiT: Magnitude-Preserving Diffusion Transformer
This project builds upon key concepts from the following research papers:
 - [Peebles & Xie (2023)](https://arxiv.org/abs/2212.09748) explore the application of transformer architectures to diffusion models, achieving state-of-the-art performance on various generation tasks;
 - [Karras et al. (2024)](https://arxiv.org/abs/2312.02696) introduce the idea of preserving the magnitude of features during the diffusion process, enhancing the stability and quality of generated outputs.

## Preliminary Results

Below, we present some preliminary results of using magnitude preservation (right) _vs._ not using magnitude preservation (left) with DiT-S/2 on the ImageNet-128 dataset. Note that DiT-S/2 is a very small model, so the samples are not of high quality. However, MaP-DiT displays much higher quality and consistency than vanilla DiT.

<p align="center">
  <img src="visuals/nomp_s_17.png" />
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="visuals/mp_s_17.png" />
  <p align="center"><b>Fig 1.</b> DiT-S/2 samples of <em>Jay</em> without (left) and with (right) magnitude preserving layers.</p>
</p>

<p align="center">
  <img src="visuals/nomp_s_88.png" />
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="visuals/mp_s_88.png" />
  <p align="center"><b>Fig 2.</b> DiT-S/2 samples of <em>Macaw</em> without (left) and with (right) magnitude preserving layers.</p>
</p>

<p align="center">
  <img src="visuals/nomp_s_247.png" />
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="visuals/mp_s_247.png" />
  <p align="center"><b>Fig 3.</b> DiT-S/2 samples of <em>St. Bernard</em> without (left) and with (right) magnitude preserving layers.</p>
</p>

<p align="center">
  <img src="visuals/nomp_s_947.png" />
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="visuals/mp_s_947.png" />
  <p align="center"><b>Fig 4.</b> DiT-S/2 samples of <em>Mushroom</em> without (left) and with (right) magnitude preserving layers.</p>
</p>

## Training

```bash
python train.py --data-path /path/to/data --results-dir /path/to/results --model DiT-S/2 --num-steps 400_000 <map feature flags>
```

## Sampling

```bash
python sample.py --result-dir /path/to/results/<dir> --class-label <class label>
```
