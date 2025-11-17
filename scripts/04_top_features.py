import sys
import json
from pathlib import Path 

import numpy as np 
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.config import load_config

def extract_top_features(model_path: Path, top_n: int = 30):
	pipe = joblib.load(model_path)

	tfidf = pipe.named_steps.get('tfidf', None)
	clf = pipe.named_steps.get('clf', None)

	if tfidf is None or clf is None:
		raise ValueError('Expercted pipeline with steps tfidf and cfg')
	

	feature_names = np.array(tfidf.get_feature_names_out())

	coefs = clf.coef_[0]

	top_pos_idx = np.argsort(coefs)[::-1][:top_n]
	top_neg_idx = np.argsort(coefs)[:top_n]

	top_positive = [
		{'feature': feature_names[i], 'weight': float(coefs[i])}
		for i in top_pos_idx
	]
	top_negative = [
		{'feature': feature_names[i], 'weight': float(coefs[i])}
		for i in top_neg_idx
	]

	return top_positive, top_negative

def main():
	cfg = load_config()
	reports_dir = Path('reports')
	reports_dir.mkdir(exist_ok=True)

	model_name = cfg.get('eval', {}).get('top_features_model', 'models/lr_fold0.joblib')
	model_path = Path(model_name)

	if not model_path.exists():
		raise SystemExit('Model file not found')

	top_n = int(cfg.get('eval', {}).get('top_n_features', 30))

	top_positive, top_negative = extract_top_features(model_path, top_n=top_n)

	print(f'Using model: {model_path}')
	print(f'\nTop {top_n} POSITIVE features (lowest weights):')
	for i, item in enumerate(top_positive, 1):
		print(f"{i:2d}. {item['feature']:<20} weight={item['weight']:4f}")

	print(f'\nTop {top_n} NEGATIVE features (lowest weights):')
	for i, item in enumerate(top_negative, 1):
		print(f"{i:2d}. {item['feature']:<20} weight={item['weight']:4f}")


	out_path = reports_dir / 'top_features.json'
	with out_path.open('w', encoding='utf-8') as f:
		json.dump(
			{
				'model_path': str(model_path),
				'top_n': top_n,
				'top_positive': top_positive,
				'top_negative': top_negative,
			},
			f,
			indent=2,
			ensure_ascii=False,)

		print (f'Saved top features to {out_path}')

if __name__ == '__main__':
	main()


























