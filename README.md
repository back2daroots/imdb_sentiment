# IMDb Sentiment Analysis — NLP Baseline Pipeline

A clean, reproducible ML pipeline for Sentiment Analysis on the IMDb movie reviews dataset.  
This project demonstrates a script-first workflow with configuration files, reproducible CV folds, OOF predictions, and baseline evaluation.

## 📌 Project Overview

This project explores sentiment analysis on the IMDb movie reviews dataset.
The goal is to compare classical NLP approaches with transformer-based models,
and to investigate how ensembling and conditional blending can improve performance.

We start with a TF-IDF + Logistic Regression baseline, move to a DistilBERT model,
and then combine both using global and conditional blending strategies.

---
## 📦 Project Structure
```
imdb/
├── configs/                 # YAML configs
│   └── base.yaml
├── data/                    # Train / test splits
├── models/                  # Saved models and OOF predictions
├── reports/                 # Metrics, confusion matrices, error analysis
├── scripts/                 # Training, evaluation and analysis scripts
│   ├── 02_train_tfidf_lr.py
│   ├── 03_eval.py
│   ├── 05_train_distilbert.py
│   ├── 06_eval_distilbert.py
│   ├── 07_blend_models.py
│   ├── 08_error_analysis.py
│   └── 09_conditional_blend.py
├── src/
│   ├── config.py
│   └── experiment_logger.py
└── README.md
```
---

## 📥 Dataset

The project accepts one of the following formats inside `data/`:

- `imdb.csv` — with columns `review`/`text` and `sentiment`/`label`  
  *(this is the format currently used)*  
- or `train.csv` + `test.csv`
- or classic IMDb folder structure:
train/pos&neg
test/pos&neg
---
## 🧪 Pipeline Overview
- Automatically detects dataset format (imdb.csv, train/test CSV, or IMDb folders) and builds:
	•	data/train.csv
	•	data/test.csv
	•	data/cv_folds.json (reproducible stratified K-Fold splits)
- Trains 5 fold models, computes OOF predictions, and stores models + metrics.
- Loads models, ensembles them (mean of probabilities), computes metrics, and saves a confusion matrix.

---

## 🤗 Transformer Baseline — DistilBERT

In addition to the TF-IDF + Logistic Regression baseline, we trained a transformer-based model using:

- **distilbert-base-uncased**
- **same 5-fold CV splits** (`cv_folds.json`)
- fine-tuning performed on **Google Colab (A100 GPU)**

Training script:
python -m scripts.05_train_distilbert

Evaluation:
python -m scripts.06_eval_distilbert

---

## 🔥 Comparison: TF-IDF Baseline vs DistilBERT

| Model                       | F1     | ROC-AUC | Accuracy |
|-----------------------------|--------|---------|----------|
| **TF-IDF + Logistic Reg.**  | 0.9161 | 0.9742  | 0.9141   |
| **DistilBERT (transformer)**| **0.9216** | **0.9765** | **0.9207** |

DistilBERT outperforms the TF-IDF baseline across all major metrics:
- +0.5% F1
- +0.6% Accuracy
- +0.23% ROC-AUC

This confirms the correctness of the training pipeline and demonstrates the expected benefit of a transformer-based approach.

---
## 🧩 Blending: TF-IDF + DistilBERT

We blended TF-IDF+LogReg and DistilBERT probabilities:

p_blend = (1 - alpha) * p_tfidf + alpha * p_distilbert

The weight \(\alpha\) was selected by maximizing **OOF F1** (using the same CV folds).

Run:
```bash
python -m scripts.07_blend_models
```

📈 Blend — Test Metrics

- F1:       0.9369
- ROC-AUC:  0.9815
- Accuracy: 0.9357

This blended model substantially outperforms both individual models.

---
## 🏆 Model Comparison (Test Set)

| Model                         | F1     | ROC-AUC | Accuracy |
|------------------------------|--------|---------|----------|
| TF-IDF + Logistic Regression | 0.9141 | 0.9742  | 0.9141   |
| DistilBERT                   | 0.9216 | 0.9765  | 0.9207   |
| Global Blend                 | 0.9369 | 0.9815  | 0.9357   |
| Conditional Blend (length)   | 0.9390 | 0.9819  | 0.9380   |

---

🕵️ Error Analysis Summary

We compared predictions across TF-IDF, DistilBERT, and the blend:
- DistilBERT improves especially on long, descriptive, context-dependent reviews, where sentiment is expressed implicitly and requires understanding the overall tone and argument structure.
- TF-IDF remains strong on short and emotionally explicit reviews, where single keywords (e.g., “awful”, “excellent”) carry most of the signal.
- Both models struggle with sarcasm/irony, mixed-sentiment reviews, and potential label noise.

As a result, blending benefits from complementary strengths and achieves the best overall quality.

---

## 🧠 Conditional Blending (by review length)

We applied a simple rule-based blend depending on review length (word count):

- If `len_words(text) <= N`: use `alpha_short`
- Else: use `alpha_long`

Parameters were selected by maximizing **OOF F1**.

Best OOF parameters:
- N = 200 words
- alpha_short = 0.50
- alpha_long = 0.20

📈 Conditional Blend — Test Metrics

- F1:       0.9390
- ROC-AUC:  0.9819
- Accuracy: 0.9380

Run:
```bash
python -m scripts.09_conditional_blend
```

---

## 🎯 Next Steps

- **Further transformer tuning** (epochs, lr, max_length, scheduler)

---

## 🔑 Key Takeaways

- Classical TF-IDF models remain strong on short and emotionally explicit texts.
- Transformer models better capture contextual and implicit sentiment.
- Blending complementary models yields significant gains without retraining.
- Simple rule-based conditional blending can further improve performance.

------


## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Win: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
