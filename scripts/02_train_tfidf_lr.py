import os, sys, json
from pathlib import Path 
import numpy as np 
import pandas as pd 


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.config import load_config

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import joblib

def ensure_dirs():
	Path('models').mkdir(exist_ok=True)
	Path('reports').mkdir(exist_ok=True)

def main():
	cfg = load_config()
	seed = int(cfg.get('seed', 42))
	tfidf_cfg = cfg.get('model', {}).get('tfidf', {})
	lr_cfg = cfg.get('model', {}).get('logreg', {})
	threshold = float(cfg.get('eval', {}).get('threshold', 0.5))

	train = pd.read_csv('data/train.csv')
	folds = json.load(open('data/cv_folds.json', 'r', encoding='utf-8'))

	texts = train['text'].astype(str).tolist()
	y = train['label'].astype(int).to_numpy()

	oof = np.zeros(len(train), dtype=float)
	per_fold = []

	ensure_dirs()

	for k, f in enumerate(folds):
		tr_idx = np.array(f['train_idx'], dtype=int)
		va_idx = np.array(f['valid_idx'], dtype=int)

		X_tr = [texts[i] for i in tr_idx]
		y_tr = y[tr_idx]
		X_va = [texts[i] for i in va_idx]
		y_va = y[va_idx]

		pipe = Pipeline([
			('tfidf', TfidfVectorizer(
				max_features=int(tfidf_cfg.get('max_features', 100000)),
				ngram_range=tuple(tfidf_cfg.get('ngram_range', [1, 2])),
				min_df=int(tfidf_cfg.get('min_df', 2)),
				sublinear_tf=bool(tfidf_cfg.get('sublinear_tf', True)),
				)),
			('clf', LogisticRegression(
				C=float(lr_cfg.get('C', 4.0)),
				max_iter=int(lr_cfg.get('max_iter', 2000)),
				n_jobs=int(lr_cfg.get('n_jobs', -1)),
				random_state=seed + k
				))
		])

		pipe.fit(X_tr, y_tr)

		proba_va = pipe.predict_proba(X_va)[:, 1]
		oof[va_idx] = proba_va

		preds_va = (proba_va >= threshold).astype(int)
		fold_metrics = {
		'fold': k,
		'f1': float(f1_score(y_va, preds_va)),
		'roc_auc': float(roc_auc_score(y_va, proba_va)),
		'accuracy': float(accuracy_score(y_va, preds_va)),
		}
		per_fold.append(fold_metrics)

		joblib.dump(pipe, f'models/lr_fold{k}.joblib')
		print(f'[fold {k}] f1={fold_metrics["f1"]:.4f} roc_auc{fold_metrics["roc_auc"]:.4f} acc={fold_metrics["accuracy"]:.4f}')


	preds_oof = (oof >= threshold).astype(int)
	metrics = {
	'oof_f1': float(f1_score(y, preds_oof)),
	'oof_roc_auc': float(roc_auc_score(y, oof)),
	'oof_accuracy': float(accuracy_score(y, preds_oof)),
	'threshold': threshold,
	'per_fold': per_fold,
	'tfidf': tfidf_cfg,
	'logrteg': lr_cfg,
	'seed': seed,
	}

	np.save('models/oof_tfidf_lr.npy', oof)
	with open('reports/cv_metrics.json', 'w', encoding='utf-8') as f:
		json.dump(metrics, f, indent=2, ensure_ascii=False)

	print('\m=== OOF metrics ==')
	print(f'F1: {metrics["oof_f1"]:.4f}')
	print(f'ROC_AUC: {metrics["oof_roc_auc"]:.4f}')
	print(f'Accuracy: {metrics["oof_accuracy"]:.4f}')
	print('Artifacts saved in models/ and reports/.')


if __name__ == '__main__':
	main()

















