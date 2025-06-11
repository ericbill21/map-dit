# Exploring Magnitude Preservation and Rotation Modulation in Diffusion Transformers
## [📝Read the paper on arXiv](https://arxiv.org/abs/2505.19122)
Denoising diffusion models exhibit remarkable generative capabilities, but remain challenging to train due to their inherent stochasticity, where high-variance gradient estimates lead to slow convergence. Previous works have shown that magnitude preservation helps with stabilizing training in the U-net architecture. This work explores whether this effect extends to the Diffusion Transformer (DiT) architecture. As such, we propose a magnitude-preserving design that stabilizes training without normalization layers. Motivated by the goal of maintaining activation magnitudes, we additionally introduce rotation modulation, which is a novel conditioning method using learned rotations instead of traditional scaling or shifting. Through empirical evaluations and ablation studies on small-scale models, we show that magnitude-preserving strategies significantly improve performance, notably reducing FID scores by $`\sim`$12.8\%. Further, we show that rotation modulation combined with scaling is competitive with AdaLN, while requiring $`\sim`$5.4\% fewer parameters. This work provides insights into conditioning strategies and magnitude control.

<p align="center">
  <img src="visuals/A-S-4.png" />
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="visuals/E-S-4.png" />
  <p align="center"><b>Fig 1.</b> DiT-S/4 samples without (left) and with (right) magnitude preserving layers.</p>
</p>

This project builds upon key concepts from the following research papers:
 - [Peebles & Xie (2023)](https://arxiv.org/abs/2212.09748) explore the application of transformer architectures to diffusion models, achieving state-of-the-art performance on various generation tasks;
 - [Karras et al. (2024)](https://arxiv.org/abs/2312.02696) introduce the idea of preserving the magnitude of features during the diffusion process, enhancing the stability and quality of generated outputs.

<!-- ## Preliminary Results

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
</p> -->

# 🚧 Code Status: Work in Progress
We're actively developing this repo. Contributions and feedback are welcome!

## Training

```bash
python train.py --data-path /path/to/data --results-dir /path/to/results --model DiT-S/2 --num-steps 400_000 <map feature flags>
```

### Magnitude Preservation Flags
Customize the training process by enabling the following flags:
- `--use-cosine-attention` - Controls weight growth in attention layers.
- `--use-weight-normalization` - Applies magnitude preservation in linear layers.
- `--use-forced-weight-normalization` - Controls weight growth in linear layers.
- `--use-mp-residual` - Enables magnitude preservation in residual connections.
- `--use-mp-silu` - Uses a magnitude-preserving version of SiLU nonlinearity.
- `--use-no-layernorm` - Disables transformer layer normalization.
- `--use-mp-pos-enc` - Activates magnitude-preserving positional encoding.
- `--use-mp-embedding` - Uses magnitude-preserving embeddings.

## Sampling

```bash
python sample.py --result-dir /path/to/results/<dir> --class-label <class label>
```

## Citation

```bibtex
@misc{bill2025exploringmagnitudepreservationrotation,
      title={Exploring Magnitude Preservation and Rotation Modulation in Diffusion Transformers}, 
      author={Eric Tillman Bill and Cristian Perez Jensen and Sotiris Anagnostidis and Dimitri von Rütte},
      year={2025},
      eprint={2505.19122},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.19122}, 
}
```
