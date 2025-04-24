# Multilingual Toxic Comment Classification with DistilBERT

## Overview

This project leverages DistilBERT, a distilled version of BERT, to classify toxic comments in a multilingual context. The goal is to accurately identify toxic (negative) comments across multiple languages using state-of-the-art transformer-based natural language processing.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Project Motivation

Online platforms receive millions of comments daily in various languages. Detecting toxic comments is essential for maintaining healthy discussions and protecting users. This project demonstrates how to use DistilBERT for efficient, scalable, and multilingual toxic comment classification.

---

## Dataset

**Source:** [Kaggle Jigsaw Multilingual Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/data)

- **Training:** `jigsaw-toxic-comment-train.csv` (English Wikipedia talk page comments)
- **Validation:** `validation.csv` (Multilingual comments)
- **Test:** `test.csv` (Multilingual comments)

**Columns:**
- `comment_text` or `content`: The comment to classify
- `toxic`: Label (1 for toxic, 0 for non-toxic)

> **Note:**  
> You must download the dataset from Kaggle and import it into your working directory before running the code.  
> The dataset is not included in this repository due to licensing restrictions.

---

## Model Architecture

- **Backbone:** `distilbert-base-multilingual-cased`  
  - Supports 104 languages
  - 60% faster and 40% smaller than BERT-base
- **Fine-tuning:**  
  - Classification head with two outputs (toxic, non-toxic)
  - Class imbalance handled with weighted loss
  - Training and evaluation performed with TensorFlow and Hugging Face Transformers

---

## Setup & Installation

1. **Clone the repository**
    ```
    git clone https://github.com/yourusername/multilingual-toxic-comment-distilbert.git
    cd multilingual-toxic-comment-distilbert
    ```

2. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

3. **Download and import the dataset from Kaggle**


---

## Usage

1. **Train the model**
    ```
    python train.py
    ```
    - You can also run the provided Jupyter notebook for step-by-step exploration.

2. **Evaluate the model**
    - Validation metrics (ROC-AUC, F1-score, etc.) are printed after training.

3. **Inference**
    - Use the `predict_toxicity()` function in the script to classify new comments.

---

## Results

- **Validation ROC-AUC:** ~0.95+
- **Cross-lingual performance:**  
    - High accuracy on English, Spanish, Turkish, Russian, and Italian comments
- **Efficient inference:**  
    - Suitable for large-scale and real-time applications

---

## File Structure

