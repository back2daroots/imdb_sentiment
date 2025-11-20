import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.config import load_config  # noqa: E402

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def make_fold_dataset(df: pd.DataFrame, train_idx, valid_idx, text_col="text", label_col="label"):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    train_ds = Dataset.from_pandas(train_df[[text_col, label_col]])
    valid_ds = Dataset.from_pandas(valid_df[[text_col, label_col]])
    return train_ds, valid_ds


def tokenize_function(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    from sklearn.metrics import f1_score, accuracy_score

    return {
        "f1": f1_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
    }


def main():
    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    data_cfg = cfg.get("data", {})
    trf_cfg = cfg.get("model", {}).get("transformer", {})

    model_name = trf_cfg.get("model_name", "distilbert-base-uncased")
    max_length = int(trf_cfg.get("max_length", 256))
    train_bs = int(trf_cfg.get("train_batch_size", 16))
    eval_bs = int(trf_cfg.get("eval_batch_size", 32))
    lr = float(trf_cfg.get("learning_rate", 2e-5))
    num_train_epochs = float(trf_cfg.get("num_train_epochs", 3))
    weight_decay = float(trf_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(trf_cfg.get("warmup_ratio", 0.1))
    grad_accum = int(trf_cfg.get("gradient_accumulation_steps", 1))

    out_root = Path("models")
    reports_root = Path("reports")
    out_root.mkdir(exist_ok=True)
    reports_root.mkdir(exist_ok=True)

    # load data and folds
    train_df = pd.read_csv("data/train.csv")
    folds = json.load(open("data/cv_folds.json", "r", encoding="utf-8"))

    texts = train_df["text"].astype(str).tolist()
    y = train_df["label"].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    oof = np.zeros(len(train_df), dtype=float)
    per_fold = []

    for k, f in enumerate(folds):
        print(f"\n==== Fold {k} ====")
        train_idx = np.array(f["train_idx"], dtype=int)
        valid_idx = np.array(f["valid_idx"], dtype=int)

        train_ds, valid_ds = make_fold_dataset(train_df, train_idx, valid_idx)

        # tokenize datasets
        train_ds = train_ds.map(
            lambda ex: tokenize_function(ex, tokenizer, max_length),
            batched=True,
        )
        valid_ds = valid_ds.map(
            lambda ex: tokenize_function(ex, tokenizer, max_length),
            batched=True,
        )

        train_ds = train_ds.rename_column("label", "labels")
        valid_ds = valid_ds.rename_column("label", "labels")
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )

        fold_out_dir = out_root / f"transformer_fold{k}"
        args = TrainingArguments(
            output_dir=str(fold_out_dir),
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            learning_rate=lr,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            #warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=grad_accum,
            #evaluation_strategy="epoch",
            #save_strategy="epoch",
            #load_best_model_at_end=True,
            #metric_for_best_model="f1",
            #greater_is_better=True,
            logging_steps=50,
            seed=seed + k,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
        )

        trainer.train()

        
        preds = trainer.predict(valid_ds)
        logits = preds.predictions
        
        probs = (logits[:, 1] - logits[:, 0])
        
        import torch
        with torch.no_grad():
            softmax_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

        oof[valid_idx] = softmax_probs

        
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

        y_va = y[valid_idx]
        pred_labels = (softmax_probs >= 0.5).astype(int)
        fold_metrics = {
            "fold": k,
            "f1": float(f1_score(y_va, pred_labels)),
            "roc_auc": float(roc_auc_score(y_va, softmax_probs)),
            "accuracy": float(accuracy_score(y_va, pred_labels)),
        }
        per_fold.append(fold_metrics)
        print(
            f"[fold {k}] f1={fold_metrics['f1']:.4f} "
            f"roc_auc={fold_metrics['roc_auc']:.4f} "
            f"acc={fold_metrics['accuracy']:.4f}"
        )

    
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
    preds_oof = (oof >= threshold).astype(int)
    oof_f1 = float(f1_score(y, preds_oof))
    oof_roc_auc = float(roc_auc_score(y, oof))
    oof_acc = float(accuracy_score(y, preds_oof))

    metrics = {
        "oof_f1": oof_f1,
        "oof_roc_auc": oof_roc_auc,
        "oof_accuracy": oof_acc,
        "threshold": threshold,
        "per_fold": per_fold,
        "transformer": trf_cfg,
        "seed": seed,
    }

    np.save(out_root / "oof_distilbert.npy", oof)
    with (reports_root / "cv_metrics_distilbert.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n== OOF metrics (DistilBERT) ==")
    print(f"F1:       {oof_f1:.4f}")
    print(f"ROC-AUC:  {oof_roc_auc:.4f}")
    print(f"Accuracy: {oof_acc:.4f}")
    print("Artifacts saved: models/oof_distilbert.npy, reports/cv_metrics_distilbert.json")


if __name__ == "__main__":
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    main()