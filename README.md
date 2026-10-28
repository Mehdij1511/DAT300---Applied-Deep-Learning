# DAT300 - Applied Deep Learning

[![Course](https://img.shields.io/badge/Course-DAT300-blue)](https://github.com/Mehdij1511/DAT300---Applied-Deep-Learning)

A portfolio-style deep learning project built as part of DAT300 - Applied Deep Learning. It brings together applied exercises and visual experiments across image classification, semantic segmentation, generative modeling, and sentiment analysis.

![Project overview](docs/figures/project_overview.png)

## Table of Contents
- [About this repo](#about-this-repo)
- [The problem this project solves](#the-problem-this-project-solves)
- [How it solves it](#how-it-solves-it)
- [What we learned](#what-we-learned)
- [Technology summary](#technology-summary)
- [Quick start](#quick-start)
- [Visualizations](#visualizations)
- [How to contribute](#how-to-contribute)
- [License](#license)

## About this repo

This repository contains course materials and student assignments for DAT300 - Applied Deep Learning. It is organized as a practical learning space where students can open assignments, run experiments locally, and generate visual explanations of how deep learning models learn.

The goal is not only to complete the course assignment, but to understand:
- how different deep learning architectures work,
- how to inspect model behavior,
- how to diagnose training problems,
- and how to communicate results in an interpretable way.

This project turns theoretical concepts into hands-on work using real model-building patterns from modern deep learning practice.

---

## The problem this project solves

In applied machine learning, the hardest part is often not “can I write a model?” but:
- How do I know if the model is learning the right thing?
- Why is the model failing on some inputs?
- What does the model actually see?
- How can I explain or visualize its decisions?
- How do I move from theory to a working pipeline?

This repository solves that by teaching deep learning through a set of focused, practical assignments:

- CNNs for image understanding
- U-Net for pixel-level segmentation
- VAEs for latent representation learning
- Sentiment analysis for text-based classification

The central theme is:
> Deep learning is not just about accuracy — it is about understanding, diagnosing, visualizing, and improving.

![Deep learning workflow](docs/figures/deep_learning_workflow.png)

---

## How it solves it

The project is structured around practical tasks, each showing a different facet of deep learning.

### 1. CNNs: learning visual features
The first assignment focuses on convolutional neural networks and how they learn spatial structures in images.

This includes:
- feature extraction,
- filter inspection,
- activation visualization,
- and understanding how early layers detect edges, shapes, and textures.

![CNN filter map](docs/figures/conv_filters.png)

This helps the learner see that CNNs do not operate like black boxes. They transform raw pixels into increasingly meaningful representations.

### 2. U-Net segmentation: localizing structure
The segmentation task shows how model outputs can be more detailed than simple classification labels. Instead of predicting a single class, the model predicts a mask for each pixel.

This teaches:
- dense prediction,
- spatial reasoning,
- boundary detection,
- and the importance of preserving detail in structured tasks.

![U-Net segmentation overlay](docs/figures/seg_overlay.png)

### 3. VAEs: learning latent representations
Variational Autoencoders are used to compress inputs into a lower-dimensional latent space and reconstruct them from that compressed representation.

This helps illustrate:
- representation learning,
- generative modeling,
- data compression,
- and the relationship between structure and reconstruction quality.

![VAE reconstructions](docs/figures/vae_recon.png)

### 4. Sentiment analysis: learning from text
The text-based assignment explores how models can classify sentiment from word patterns and language structure.

This introduces:
- text preprocessing,
- feature extraction,
- class-based interpretation,
- and the value of visual analysis through word clouds and frequency summaries.

![Sentiment wordcloud](docs/figures/wordcloud.png)

---

## What we learned

This project is not just about implementing architectures. It is about learning how to reason with models.

### Key lessons
- Deep learning depends on data quality, preprocessing, and architecture choices.
- Training curves reveal more than accuracy alone.
- Model inspection matters just as much as model performance.
- Visualizations can reveal bias, underfitting, overfitting, or poor representation.
- Generative models teach us about structure and compression.
- Segmentation demands spatial understanding.
- Text models require careful treatment of language structure and interpretation.

A model is not “good” just because it reaches a high score. It is useful when we can explain how it behaves, where it fails, and what it is learning.

![Training curves](docs/figures/training.png)

---

## Technology summary

This project uses a practical Python-based deep learning stack suitable for education and experimentation.

Core technologies:
- Python
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Jupyter / JupyterLab
- PyTorch
- TensorFlow / Keras (depending on assignment setup)
- scikit-learn
- Pillow
- WordCloud

These tools form a standard workflow for:
- data preparation,
- model training,
- evaluation,
- visualization,
- and interpretation.

![Technology stack](docs/figures/tech_stack.png)

---

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/Mehdij1511/DAT300---Applied-Deep-Learning.git
cd DAT300---Applied-Deep-Learning
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

4. Launch Jupyter:

```bash
pip install jupyterlab
jupyter lab
```

5. Open a notebook or script and reproduce the visualizations or experiments.

To generate the figures locally from the repository code:

```bash
python scripts/generate_figures.py
```

The generator creates illustrative sample figures in `docs/figures/`; install
the required plotting packages with `pip install -r requirements.txt`.

---

## Visualizations

The project includes example visual outputs that help explain what the model is doing.

### Training curves (loss & accuracy)
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

def plot_training(history, out_path='docs/figures/training.png'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
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
```python
import matplotlib.pyplot as plt
import numpy as np

def show_filters(weight_tensor, n_cols=8, out_path='docs/figures/conv_filters.png'):
    filters = weight_tensor.cpu().numpy()
    n = min(filters.shape[0], 64)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    for i in range(n):
        ax = axes.flat[i]
        f = filters[i]
        img = f.mean(axis=0)
        ax.imshow((img - img.min()) / (img.max() - img.min()), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

### U-Net segmentation overlay
```python
import matplotlib.pyplot as plt

def overlay_segmentation(image, mask, alpha=0.5, out_path='docs/figures/seg_overlay.png'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.imshow(mask, cmap='Reds', alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
```

### VAE reconstructions
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_reconstructions(x, x_recon, n=8, out_path='docs/figures/vae_recon.png'):
    plt.figure(figsize=(2 * n, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow((x[i] * 255).astype('uint8'))
        plt.axis('off')
        plt.subplot(2, n, n + i + 1)
        plt.imshow((x_recon[i] * 255).astype('uint8'))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
```

### Sentiment wordclouds
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def save_wordcloud(text, out_path='docs/figures/wordcloud.png'):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
```

---

## How to contribute

- Add runnable notebooks to `notebooks/` that reproduce the figures and experiments in the assignments.
- Add scripts to `src/` and helpers to `utils/`.
- Save visual outputs to `docs/figures/` and reference them here.
- Keep examples reproducible and easy to extend.
- Do not commit large datasets; prefer small fixtures or public sample data.
- Open a pull request with a clear explanation of what changed.

---

## License

This repository is intended for educational and course-related use. Check with the course owner for licensing details.

![Applied deep learning workflow](docs/figures/applied_workflow.png)
