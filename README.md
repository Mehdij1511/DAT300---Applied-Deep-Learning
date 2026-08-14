# DAT300 - Applied Deep Learning

[![Course](https://img.shields.io/badge/Course-DAT300-blue)](https://github.com/Mehdij1511/DAT300---Applied-Deep-Learning)

DAT300 builds on DAT200 and covers practical deep learning techniques used for modern ML problems: convolutional neural networks, recurrent networks, autoencoders, variational autoencoders, basic GANs, U-Net segmentation, and transformer introductions.

## Table of Contents
- [About this repo](#about-this-repo)
- [Contents](#contents)
- [Quick start](#quick-start)
- [Visualizations & Examples](#visualizations--examples)
  - [Training curves (loss & accuracy)](#training-curves-loss--accuracy)
  - [CNN filters & feature maps](#cnn-filters--feature-maps)
  - [U-Net segmentation overlays](#u-net-segmentation-overlays)
  - [VAE reconstructions](#vae-reconstructions)
  - [Sentiment analysis wordclouds](#sentiment-analysis-wordclouds)
- [Suggested requirements](#suggested-requirements)
- [How to contribute](#how-to-contribute)
- [License](#license)

## About this repo
This repository contains course materials and student assignments for DAT300 - Applied Deep Learning. It is organized so you can open the assignments, run experiments locally, and generate the visualizations commonly used to inspect model behaviour.

## Contents
- CA2-ConvolutionalNeuralNetworks.pdf — Assignment and write-up for CNN exercises
- CA3-SegmanticSegmentationU-net.pdf — U-Net segmentation assignment
- CA4-VariationalAutoencoders.pdf — VAE assignment
- CA5-SentimentAnalasys.pdf — Sentiment analysis assignment
- README.md — this file

## Quick start
1. Clone the repo:

```bash
git clone https://github.com/Mehdij1511/DAT300---Applied-Deep-Learning.git
cd DAT300---Applied-Deep-Learning
```

2. Create a virtual environment and install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -U pip
pip install -r requirements.txt  # see Suggested requirements below
```

3. Open a notebook or script to reproduce visualizations. For example:

```bash
pip install jupyterlab
jupyter lab
# then open a notebook that loads the assignment datasets and code
```

Notes: this repo currently contains PDFs of assignments. If you (or the instructor) add notebooks or scripts, place them under `notebooks/` or `src/` and add figures to `docs/figures/`.

## Visualizations & Examples
Below are short, copy-pasteable examples for common visualizations used in the assignments. They are small snippets intended to be dropped into a notebook or script. Save resulting figures to `docs/figures/` and commit them so the README can display them.

### Training curves (loss & accuracy)
Save training history (loss/accuracy per epoch) and plot:

```python
# name: utils/plot_training.py
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

def plot_training(history, out_path='docs/figures/training.png'):
    # history: dict with 'loss','val_loss','accuracy','val_accuracy' lists
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].plot(history['loss'], label='train')
    if 'val_loss' in history:
        axs[0].plot(history['val_loss'], label='val')
    axs[0].set_title('Loss')
    axs[0].legend()

    if 'accuracy' in history:
        axs[1].plot(history['accuracy'], label='train')
        if 'val_accuracy' in history:
            axs[1].plot(history['val_accuracy'], label='val')
        axs[1].set_title('Accuracy')
        axs[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

### CNN filters & feature maps
Inspect convolutional filters and intermediate feature maps to understand what the network learns:

```python
# inspect first-layer filters (PyTorch example)
import matplotlib.pyplot as plt
import numpy as np

def show_filters(weight_tensor, n_cols=8, out_path='docs/figures/conv_filters.png'):
    # weight_tensor: (out_channels, in_channels, H, W)
    filters = weight_tensor.cpu().numpy()
    n = min(filters.shape[0], 64)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    for i in range(n):
        ax = axes.flat[i]
        f = filters[i]
        # if RGB (in_channels==3), average channels for display
        img = f.mean(axis=0)
        ax.imshow((img - img.min()) / (img.max() - img.min()), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

### U-Net segmentation overlays
Show predicted mask over original image for quick qualitative inspection:

```python
import matplotlib.pyplot as plt

def overlay_segmentation(image, mask, alpha=0.5, out_path='docs/figures/seg_overlay.png'):
    # image: HxW or HxWx3, mask: HxW boolean or 0/1
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray' if image.ndim==2 else None)
    plt.imshow(mask, cmap='Reds', alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
```

### VAE reconstructions
Plot original and reconstructed images side-by-side to check latent quality:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_reconstructions(x, x_recon, n=8, out_path='docs/figures/vae_recon.png'):
    # x, x_recon: (batch, H, W, C)
    plt.figure(figsize=(2*n,4))
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow((x[i]*255).astype('uint8'))
        plt.axis('off')
        plt.subplot(2,n,n+i+1)
        plt.imshow((x_recon[i]*255).astype('uint8'))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
```

### Sentiment analysis wordclouds
Quick way to visualize most common words per class (requires wordcloud package):

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def save_wordcloud(text, out_path='docs/figures/wordcloud.png'):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
```

## Suggested requirements
Create a `requirements.txt` with the packages below (versions are suggestions — pin as needed):

```
python>=3.8
numpy
scipy
matplotlib
seaborn
jupyterlab
torch torchvision  # or tensorflow/keras depending on assignments
scikit-learn
Pillow
wordcloud
notebook
```

## How to contribute
- Add runnable notebooks to `notebooks/` that reproduce the figures in the PDFs.
- Add scripts to `src/` and helpers to `utils/`.
- Save visual outputs to `docs/figures/` and reference them in this README.
- Open a pull request; label notebooks and large data files appropriately (do not commit large datasets). If you need sample datasets, use small fixtures or point to public datasets.

## License
Check with the course owner for license. If you want, I can add an MIT license file.

----

If you’d like, I can:
- add example notebooks that generate the visualizations above,
- create a `requirements.txt` and CI workflow to run tests/docs,
- generate and commit example figures into `docs/figures/` so the README shows images.
