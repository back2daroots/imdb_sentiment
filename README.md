# IMDb Sentiment Analysis — NLP Baseline Pipeline

A clean, reproducible ML pipeline for Sentiment Analysis on the IMDb movie reviews dataset.  
This project demonstrates a script-first workflow with configuration files, reproducible CV folds, OOF predictions, and baseline evaluation.

---
## 📦 Project Structure
```
imdb/
├── configs/
│   ├── config.yaml                				 # Feature, model, and path settings
    └──  local.yaml                        # Local override
├── data/                                  # dataset (ignored by git)
├── models/                        				 # Saved models per fold (ignored by git)
├── reports/                      				 # Metrics, plots (ignored by git)
├── src/                                   # Reusable code (config loader, utils etc)
│   ├──                       	      		 #  
│   ├──                 			          	 #  
│   ├──                   			        	 #  
│   ├──                			            	 #  
│   ├──          			                     #  
│   └──                   			           # 
├── experiments_log.csv      				       # Experiment registry
├── requirements.txt                       # Dependencies specification
├── README.md         				     
└── .gitignore
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

## 📊 Current Results (OOF + Test)

OOF (train):
- F1:       0.9120
- ROC_AUC:  0.9706
- Accuracy: 0.9097

Test:
- F1:       0.9161
- ROC_AUC:  0.9742
- Accuracy: 0.9141


---

## ▶️ Next Steps
	•	04_top_features.py: interpret LR weights (top positive/negative words)
	•	Transformer baseline (DistilBERT) using same CV folds + OOF
	•	Model blending TF-IDF + Transformer
	•	Error analysis




## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Win: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
