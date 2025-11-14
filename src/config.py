from __future__ import annotations
import argparse
import os
from typing import Any, Dict

import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config loader")
    p.add_argument("--base", default="configs/base.yaml", help="Path to base config")
    p.add_argument("--local", default="configs/local.yaml", help="Path to local override")

    p.add_argument("--override", nargs="*", default=[], help="Dotlist overrides: key=value (e.g., model.logreg.C=2.0)")
    return p.parse_args()

def apply_dot_overrides(cfg: Dict[str, Any], pairs: list[str]) -> Dict[str, Any]:
    for pair in pairs:
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)

        casted: Any = val
        if val.lower() in ("true", "false"):
            casted = val.lower() == "true"
        else:
            try:
                if "." in val:
                    casted = float(val)
                else:
                    casted = int(val)
            except ValueError:
                pass

        cur = cfg
        *parents, last = key.split(".")
        for k in parents:
            cur = cur.setdefault(k, {})
        cur[last] = casted
    return cfg

def load_config() -> Dict[str, Any]:
    args = parse_args()
    cfg = load_yaml(args.base)
    cfg = deep_update(cfg, load_yaml(args.local))  
    cfg = apply_dot_overrides(cfg, args.override)
    return cfg

if __name__ == "__main__":
    cfg = load_config()
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))