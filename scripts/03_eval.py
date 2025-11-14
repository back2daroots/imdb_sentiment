import os, sys, glob, json
from pathlib import Path 
import numpy as np 
import pandas as pd 
import joblib
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt 

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.config import load_config

def ensure_dirs():
    Path('reports').mkdir(exist_ok=True)

def plot_confusion(cm: np.ndarray, out_path: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (test)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["neg", "pos"])
    plt.yticks(tick_marks, ["neg", "pos"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    cfg = load_config()
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))

    ensure_dirs()

    test = pd.read_csv("data/test.csv")
    paths = sorted(glob.glob("models/lr_fold*.joblib"))
    if not paths:
        raise SystemExit("models not found")

    probas = np.zeros(len(test), dtype=float)
    for p in paths:
        m = joblib.load(p)
        probas += m.predict_proba(test["text"].astype(str).tolist())[:, 1]
    probas /= len(paths)

    preds = (probas >= threshold).astype(int)
    y_true = test["label"].astype(int).to_numpy()

    metrics = {
        "f1": float(f1_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probas)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "threshold": threshold,
        "n_models": len(paths),
    }
    with open("reports/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cls_report = classification_report(y_true, preds, output_dict=True)
    with open("reports/test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(cls_report, f, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_true, preds)
    plot_confusion(cm, "reports/confusion_matrix.png")

    print('== Test metrics==')
    print(f'F1: {metrics["f1"]:.4f}')
    print(f'ROC_AUC: {metrics["roc_auc"]:.4f}')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print('Artifacts saved in reports/.')

if __name__ == '__main__':
    main()