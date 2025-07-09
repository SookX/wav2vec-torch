# 🎧 Wav2Vec2.0 from Scratch

🧠 A full **PyTorch** implementation of **wav2vec2.0**, designed to learn powerful audio representations from raw waveform input using **contrastive learning** — built completely from scratch.

📌 This repo focuses on **unsupervised pretraining** and **embedding extraction**, not full speech recognition (ASR). Fine-tuning is in progress on **Bulgarian** 🇧🇬 speech data.

---

## 📄 Paper Reference

This work is based on the **wav2vec 2.0** paper:

> 📝 **Baevski et al., 2020** – "[wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)"  
> 📍 *Facebook AI Research (FAIR)*

This implementation aims to closely follow the core ideas of wav2vec 2.0:  
✅ Self-supervised training on raw audio  
✅ Contrastive learning with quantization  
✅ Contextualized representations via Transformer

---

## 🧠 Model Architecture

The **wav2vec 2.0** model learns contextual speech representations through self-supervised contrastive learning. It consists of the following key components:

---

### 1. 🎛️ Feature Encoder

A multi-layer convolutional network that encodes raw waveform input into latent speech representations.

- **Input**: Raw audio waveform (16 kHz)
- **Output**: Latent feature sequence $\mathbf{z} = (z_1, ..., z_T)$

---

### 2. 🎯 Quantization Module

This module discretizes the continuous latent representations into a set of learned codebook vectors. It uses **product quantization** and **Gumbel-Softmax** to make the sampling differentiable.

- Produces a discrete target $\mathbf{q}_t$ for each time step
- Enables contrastive learning without requiring labeled data

---

### 3. 🧮 Context Network (Transformer)

A stack of **transformer blocks** that model the full sequence to produce context-aware representations.

- **Input**: Latent features $\mathbf{z}$
- **Output**: Contextual embeddings $\mathbf{c} = (c_1, ..., c_T)$

---

### 4. 📉 Contrastive Loss

The model is trained to **distinguish true future samples from negatives**. For a given context vector $c_t$, it tries to correctly identify the true quantized latent target $q_t$ among a set of distractors.

The **contrastive loss** is defined as:

**L<sub>m</sub> = -log [ exp(sim(c<sub>t</sub>, q<sub>t</sub>) / κ) / Σ<sub>q̃ ∈ Q<sub>t</sub></sub> exp(sim(c<sub>t</sub>, q̃) / κ) ]**


Where:
- $\text{sim}(a, b)$ is the cosine similarity between vectors $a$ and $b$
- $\kappa$ is a temperature hyperparameter
- $\mathcal{Q}_t$ is the set containing the positive sample $q_t$ and $K$ negative samples

The final training objective includes both:
- **Contrastive loss** $\mathcal{L}_m$
- **Codebook diversity loss** $\mathcal{L}_d$ (to ensure codebook usage)
  \
**L = L_m + α · L_d**

---

### 🖼️ Architecture Diagram

<p align="center">
  <img src="https://jonathanbgn.com/assets/images/illustrated-wav2vec/wav2vec2_architecture_pretraining.png" alt="wav2vec 2.0 Architecture" width="80%">
</p>

📚 *Image credit: [Jonathan Binas – Illustrated Wav2Vec 2.0](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)*  
*Highly recommended read for a clear and intuitive walkthrough of the full architecture.*

---

## ⚙️ Pretraining

The wav2vec 2.0 model was pretrained on the **LibriSpeech 100-hour** dataset using the following configuration:

- **Model dimension:** 768  
- **Epochs:** 5  
- **Batch size:** 8  
- **Learning rate:** 3e-4  

Training was performed on a single **NVIDIA RTX 3080 GPU** with 12 GB of VRAM.  
The entire pretraining process took **≈ 12 hours**.

---

## Fine-tuning
---

## 💾 Model Checkpoints

Pretrained and fine-tuned model checkpoints are available for download:

- **Pretrained model** on LibriSpeech 100-hour dataset: [Download link coming soon]  
- **Fine-tuned model** on Bulgarian speech dataset: [Download link coming soon]

---

## 📢 Usage & License

This project is **publicly available** and **free to use** for research and personal projects.  
Feel free to clone, experiment, and build on top of this implementation!

Please note that due to hardware limitations, I have **not been able to extensively test all use cases** of this model on my local machine. 

If you encounter any issues or notice unexpected behavior with the model, please don’t hesitate to contact me.

If you use this work in your research or projects, please consider citing the original wav2vec 2.0 paper and this repository.

---



