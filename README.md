# Transformer-Based Character Language Model (Mini GPT)

![Profile Views](https://komarev.com/ghpvc/?username=tonypanda7&label=shakesphere-GPT%20Views&color=blue&style=for-the-badge)

<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
	<img src="https://img.shields.io/badge/Deep%20Learning-FF6F00.svg?style=flat" alt="Deep Learning">
	<img src="https://img.shields.io/badge/Transformer-FF8C00.svg?style=flat" alt="Transformer">
	<img src="https://img.shields.io/badge/NLP-4CAF50.svg?style=flat" alt="NLP">
	<img src="https://img.shields.io/badge/Tiny%20Shakespeare-Dataset-lightgrey.svg?style=flat" alt="Dataset">
</p>

---

## Overview

A from-scratch implementation of a decoder-only Transformer (GPT-style) using PyTorch, trained on the Tiny Shakespeare dataset for character-level text generation.  
This project demonstrates core concepts behind modern LLMs such as Self-attention, multi-head attention, and autoregressive generation.

> Input → Token Embedding → Positional Embedding → Transformer Blocks → LayerNorm → Linear Head → Output Probabilities

---

## Core Components

- Embedding Layer  
Maps characters → dense vectors (n_embed = 64)  
Learns semantic relationships between tokens  

- Positional Encoding  
Encodes sequence order using learned embeddings  
Shape: (block_size, n_embed)  

- Transformer Block (×4)  

Each block consists of:

Multi-Head Self-Attention  
Parallel attention heads  
Captures different contextual relationships  

Feed Forward Network  
Expands → non-linearity → projection  

Residual Connections  
Stabilizes training  

Layer Normalization  
Improves convergence  

---

## SETUP

Install Dependencies
```python
pip install torch
```
Run file
```python
python <filename>.py
```
