# IMDB Sentiment Analysis

A binary sentiment classifier trained on IMDB movie reviews using TensorFlow and Keras. The model predicts whether a movie review is **positive** or **negative**.

---

## Overview

This project builds a neural network pipeline that:
- Vectorizes raw text using `TextVectorization`
- Learns word embeddings from scratch
- Classifies reviews as positive or negative

Achieves ~**85–87% accuracy** on the IMDB test set.

---

## Dataset

- **Source:** [IMDB Reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews) via TensorFlow Datasets
- **Size:** 25,000 training samples, 25,000 test samples
- **Split used:** 60% train / 40% validation / test set

---

## Model Architecture

```
TextVectorization  →  Embedding(10000, 16)  →  GlobalAveragePooling1D  →  Dense(16, relu)  →  Dense(1)
```

| Layer | Details |
|---|---|
| TextVectorization | max_tokens=10000, sequence_length=250 |
| Embedding | vocab_size=10000, embedding_dim=16 |
| GlobalAveragePooling1D | Averages across token dimension |
| Dense | 16 units, ReLU activation |
| Output Dense | 1 unit, logits output |

---

## Requirements

```bash
pip install tensorflow
pip install tensorflow-datasets
```

---

## Usage

Open the notebook and run all cells in order:

```
imdb-sentiment-analysis.ipynb
```

To run inference on a custom review:

```python
import tensorflow as tf

new_reviews = ["This movie was absolutely fantastic!",
               "Terrible film, complete waste of time."]

predictions = tf.sigmoid(model.predict(new_reviews))
for review, score in zip(new_reviews, predictions):
    print(f"{'Positive' if score > 0.5 else 'Negative'} ({score[0]:.2f}) — {review}")
```

---

## Results

| Metric | Score |
|---|---|
| Train Accuracy | ~90% |
| Validation Accuracy | ~86% |
| Test Accuracy | ~85–87% |

---

## Tech Stack

- Python
- TensorFlow / Keras
- TensorFlow Datasets
- NumPy

---

