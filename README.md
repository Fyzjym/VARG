<div align="center">

<img src="assets/header.svg" width="100%" alt="VARG — Structure, style, and visual representation guidance" />

# VARG

### Structure-Aware Handwritten Mathematical Expression Generation<br>with Visual Autoregressive Representation Guidance

Yiming Wang · Heng Wang · Hongxi Wei

![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C?style=flat-square)
![Task](https://img.shields.io/badge/task-handwritten_math_generation-243B53?style=flat-square)
![Conditioning](https://img.shields.io/badge/conditioning-SAT_%2B_HCEM-007F82?style=flat-square)

[Overview](#overview) · [Architecture](#architecture) · [Quick start](#quick-start) · [Data](#data-preparation) · [Interfaces](#public-interfaces) · [Results](#manuscript-results) · [Citation](#citation)

</div>

## Overview

VARG synthesizes handwritten mathematical expressions from a **rendered content image** and a **reference handwriting image**. It combines coarse-to-fine, style-aware visual representation guidance with conditional latent diffusion to preserve two-dimensional expression structure and handwriting appearance.

The autoregressive component operates on **continuous multiscale representations**, not discrete image tokens. The final image is produced by a diffusion U-Net and a frozen VAE decoder; VARG is not an autoregressive image decoder.

> This repository contains the supplied research implementation, with paper-aligned module names and cleaned interfaces. Metrics and example images below come from the manuscript, not new runs during the interface cleanup. Datasets, trained VARG checkpoints, and evaluation recognizers are not bundled.

## Architecture

<p align="center"><img src="assets/architecture.png" width="100%" alt="VARG architecture from manuscript Figure 1" /></p>

Separate visual encoders extract handwriting style and printed content. SAT constructs a style-conditioned representation; HCEM refines it through HEU and CAM. The resulting context conditions the diffusion denoiser through cross-attention.

| Component | Full name | Role | Implementation |
| :-- | :-- | :-- | :-- |
| SAT | Style-Aware Transformer | Style-modulated multiscale attention with AdaLN and RoPE | [fusion.py](models/fusion.py) |
| HCEM | Hierarchical Content Enhancement Module | Container for HEU and CAM | [hcem.py](models/hcem.py) |
| HEU | Hyperbolic Embedding Unit | Hyperbolic processing of content features | [heu.py](models/heu.py) |
| CAM | Context Aggregation Module | Gated residual fusion of SAT and HEU features | [hcem.py](models/hcem.py) |
| VARG | Conditional diffusion network | U-Net denoising conditioned on SAT–HCEM context | [unet.py](models/unet.py) |

<details>
<summary>Inside SAT and HCEM</summary>

<p align="center"><img src="assets/sat.png" width="56%" alt="Style-Aware Transformer, manuscript Figure 2" /> <img src="assets/hcem.png" width="42%" alt="Hierarchical Content Enhancement Module, manuscript Figure 3" /></p>

SAT pools content at scales **1, 2, 4, 8, 16**, creating **341 tokens**. Each token can attend to its own scale and all coarser scales. Six transformer blocks use style-conditioned AdaLN and rotary positions. The finest **256 tokens**, projected to **512 dimensions**, provide diffusion context.

The conditioner instantiates HEU with four hyperbolic convolution blocks. CAM retains `SAT_features + sigmoid(MLP(SAT_features)) * HEU_features`.

**Implementation boundary.** This release does not rewrite numerical operations. The supplied HEU uses Möbius exponential/logarithmic-map operations, whereas the manuscript describes a Lorentz-model formulation. CAM retains its existing gated residual computation rather than introducing new normalization/projection layers from the conceptual diagram. These mathematical differences are documented, not silently changed.

</details>

## Quick start

### Environment

Use Python 3.10 or newer for a fresh environment. Training requires Linux, an NVIDIA GPU, a compatible CUDA-enabled PyTorch build, and NCCL. The manuscript reports a single NVIDIA V100 with 32 GB memory.

```bash
git clone https://github.com/Fyzjym/VARG.git
cd VARG
python -m venv .venv
source .venv/bin/activate
# Install the CUDA-enabled torch/torchvision pair appropriate for your machine first.
python -m pip install -r requirements.txt
```

The requirements target PyTorch 2.4 / torchvision 0.19. Choose the appropriate CUDA distribution for your machine. CPU interface tests do not require CUDA.

### Prepare and train

Provide a local Stable Diffusion v1.5 directory containing `vae/`. VARG loads only `AutoencoderKL` and freezes it; this is **not** a pretrained VARG checkpoint.

Prepare the images and annotations below, then update [configs/crohme.yaml](configs/crohme.yaml). `DATA_LOADER.VALIDATION_CONTENT` must point to an existing rendered expression image.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  --config configs/crohme.yaml \
  --vae-path /path/to/stable-diffusion-v1-5 \
  --run-name crohme-varg
```

The entry point initializes distributed training even for one GPU; use `torchrun`, not bare `python train.py`. The device is selected using `LOCAL_RANK`.

| Setting | Example configuration |
| :-- | :-- |
| Image canvas | 256 × 256 |
| Batch size | 32 per process; 32 total with one GPU |
| Optimizer / learning rate | AdamW / 1 × 10⁻⁴ |
| Epochs | 700 |
| Diffusion | 1,000 steps; linear β from 10⁻⁴ to 0.02 |
| Validation sampler | DDIM, 50 steps |
| Objective | Diffusion MSE + writer-supervised style contrastive loss |

These main settings follow the manuscript; the example is not a guarantee of numerical reproduction. Multi-GPU total batch size equals configured batch size × process count. `SOLVER.GRAD_L2_CLIP` applies only to the existing auxiliary fine-tuning branch, not the main training path.

```text
Saved/crohme/<run-name>-<timestamp>/
├── model/       # <zero-based-epoch>-ckpt.pt
├── sample/      # validation grids
└── tboard/      # loss events
```

The example saves and validates every 25 epochs. `--checkpoint /path/to/ckpt.pt` loads model weights, not optimizer state or epoch progress. `--encoder-checkpoint` optionally loads a separate ResNet-18 style-encoder state dictionary.

## Data preparation

The manuscript uses 8,836 CROHME training expressions from 352 writers. CROHME 2014/2016/2019 test sets contain 986/1,147/1,199 expressions. UniMER-HWE and HME100K are used for cross-dataset evaluation, not additional training in the reported protocol.

```text
data/CROHME/
├── annotations/{train.txt,test.txt,all.txt}
├── images/{train,test}/<writer-id>/<expression-id>.png
├── styles/{train,test}/<writer-id>/<reference-id>.png
└── content/{train,test}/<writer-id>/<expression-id>.png
```

Each annotation has **four fields separated by exactly two spaces**:

```text
expression-id  metadata  writer-id  space-tokenized-LaTeX
example001  unused  12  x = r \cos \theta
```

The second field is unused; expression IDs exclude `.png`, and writer IDs must parse as integers. `all.txt` is needed only by `LaTeXAnnotationCollection`.

Each writer needs at least **two style references**, sampled without replacement for the contrastive objective. Target and content images are RGB normalized to `[-1, 1]`; grayscale styles use `[0, 1]`. The existing loader resizes all inputs to 256 × 256 using nearest-neighbor interpolation and filters expressions above its default 96-token limit.

Prepare aspect-preserving padded canvases **offline** to follow the manuscript preprocessing. Raw InkML conversion and LaTeX rendering are not included.

`LAPLACE_PATH` may be empty; the adapter then supplies zero tensors in that reserved slot. Laplace images do not enter SAT or HCEM. If provided, their directory structure must match the style references.

## Public interfaces

```python
from models import VARG, VARGConditioner, SAT, HCEM, HEU, CAM
from data_loader.loader import HMEDataset, ContentImage
from utils.checkpoint import load_varg_checkpoint
```

| Interface | Contract |
| :-- | :-- |
| `VARGConditioner(d_model=512)` | Exposes `sat`, `hcem.heu`, `hcem.cam` |
| `conditioner(style, content=content)` | Two style views `[B, 2, 256, 256]`; returns context `[B, 256, 512]` and embeddings `[B, 2, 256]` |
| `conditioner.generate(style, content=content)` | Single reference `[B, 1, 256, 256]`; returns context `[B, 256, 512]` |
| `SAT(content_features, style_features)` | Default maps `[B, 512, 16, 16]`; output `[B, 256, 512]` |
| `HCEM(sat_features, content_features)` | Refines SAT tokens with HEU and CAM |
| `ContentImage(path).load()` | Normalized content `[1, 1, 256, 256]` |
| `HMEDataset(...).collate_batch(batch)` | Training adapter used by `train.py` |
| `load_varg_checkpoint(model, path)` | Strict tensor loading; migrates prior module paths and removes an optional DDP prefix |

Reserved `laplace` and `latex` arguments remain for data-adapter compatibility; neither changes SAT–HCEM conditioning. Checkpoint-name migration does not reshape or modify parameter tensors.

### Generate from an existing checkpoint

Instantiate the training configuration, load weights, and call the existing sampler; no further training is required.

```python
import torch
from diffusers import AutoencoderKL
from models import VARG
from models.diffusion import Diffusion
from utils.checkpoint import load_varg_checkpoint

device = torch.device("cuda")
model = VARG(
    in_channels=4, model_channels=512, out_channels=4,
    num_res_blocks=1, attention_resolutions=(1, 1),
    channel_mult=(1, 1), num_heads=4, context_dim=512,
).to(device)
load_varg_checkpoint(model, "/path/to/varg-ckpt.pt")
vae = AutoencoderKL.from_pretrained(
    "/path/to/stable-diffusion-v1-5", subfolder="vae"
).to(device).requires_grad_(False).eval()

# Supply prepared tensors on device:
# style [B, 1, 256, 256] in [0, 1]; content [B, 1, 256, 256] in [-1, 1].
batch_size = style.shape[0]
noise = torch.randn(batch_size, 4, 32, 32, device=device)
images = Diffusion(device=device).ddim_sample(
    model, vae, batch_size, noise, style,
    torch.zeros_like(style), content, sampling_timesteps=50,
)
# CPU float images [B, 3, 256, 256] in [0, 1].
```

This is an integration sketch: supply your tensors and local model paths. The existing sampler returns the network to training mode; call `model.eval()` afterward if needed. Visual encoders request torchvision ResNet-18 weights at construction; cache them for offline use.

## Manuscript results

**CROHME 2019 — Table 1.** ExpRate uses PosFormer; the last column uses Qwen2.5-VL-7B-Instruct. These are reported numbers, not newly reproduced results.

| FID ↓ | SSIM ↑ | HWD ↓ | ExpRate ↑ | ExpRate<sub>Qwen</sub> ↑ |
| :--: | :--: | :--: | :--: | :--: |
| 5.58 | 0.906 | 0.1459 | 34.31% | 26.71% |

**Cross-dataset evaluation — Table 9.** Training uses CROHME only.

| Dataset | FID ↓ | HWD ↓ | ExpRate ↑ |
| :-- | :--: | :--: | :--: |
| UniMER-HWE | 22.21 | 0.6591 | 10.46% |
| HME100K | 36.96 | 0.7724 | 11.73% |

Adding 2,000 synthetic expressions **together with scale augmentation** yields PosFormer ExpRate of 62.54%, 63.46%, and 63.80% on CROHME 2014/2016/2019 (Table 10). Those values describe the combined setting, not synthesis alone. Evaluation extractors and downstream recognizer training are outside this release.

<p align="center"><img src="assets/content-editing.png" width="80%" alt="Manuscript Figure 5: content editing and compositional expression ladders" /></p>
<p align="center"><sub>Content editing and compositional generalization — manuscript Figure 5.</sub></p>

## Repository map

```text
VARG/
├── assets/                 # README artwork and manuscript figures
├── configs/crohme.yaml     # editable training example
├── data_loader/loader.py   # HME targets, references, content adapters
├── models/
│   ├── fusion.py           # SAT and VARGConditioner
│   ├── hcem.py             # HCEM and CAM
│   ├── heu.py              # HEU and hyperbolic operations
│   ├── unet.py             # VARG diffusion network
│   ├── diffusion.py        # schedule, DDIM/DDPM samplers
│   └── loss.py             # supervised contrastive objective
├── trainer/trainer.py      # training, validation, checkpoints
├── tests/                  # CPU interface tests
├── utils/checkpoint.py     # checkpoint-name compatibility
├── parse_config.py         # YAML configuration loader
├── requirements.txt
└── train.py                # torchrun entry point
```

Run `python -m unittest discover -s tests -v` for interface tests. They do not replace full GPU training or generation-quality evaluation. The auxiliary OCR fine-tuning branch is not used by the supplied command and was not refactored in this release.

## Citation

Use this manuscript citation until final bibliographic details are available. No acceptance status, venue, or DOI is asserted.

```bibtex
@unpublished{wang2026varg,
  title  = {VARG: Structure-Aware Handwritten Mathematical Expression Generation
            with Visual Autoregressive Representation Guidance},
  author = {Wang, Yiming and Wang, Heng and Wei, Hongxi},
  year   = {2026},
  note   = {Manuscript}
}
```

## Acknowledgments and usage

We acknowledge PyTorch, torchvision, Hugging Face Diffusers, Stable Diffusion's latent autoencoder, and the study's datasets and recognition systems. Existing source attribution, including the GANwriting-derived adapter attribution, is retained. Weights and datasets remain subject to their respective licenses and access conditions.

The [MIT license](LICENSE) is preserved from this repository's existing `main` branch, including its copyright notice. It does not override third-party weight or dataset terms.
