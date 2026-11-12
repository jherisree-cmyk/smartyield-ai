"""Step 8 - promote the chosen feature mode's models to the canonical names
used by the app and the experiment scripts (models/yield_{track}.joblib,
outputs/yield_results_{track}.json, outputs/yield_predictions_{track}.csv).

Rule (pre-specified): per the multispectral ablation, choose the mode whose
selected model has the lower P1 grouped-CV RMSE. A mode whose run was
stopped before saving (P1 only, for CPU-time reasons) can still be compared
through the P1 lines of its training log.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartyield import config  # noqa: E402

O, M = config.OUTPUTS_DIR, config.MODELS_DIR


def best_p1(mode: str) -> float:
    lp = O / f"train_ms_{mode}.log"
    vals = [float(m[2]) for m in re.finditer(r"\[P1\] (.+?)\s+RMSE ([\d.]+)", lp.read_text())
            if "baseline" not in m[1]] if lp.exists() else []
    return min(vals) if vals else float("inf")


def main():
    scores = {m: best_p1(m) for m in ("abs", "absrel")}
    mode = min(scores, key=scores.get)
    print("P1 best RMSE per mode:", scores, "-> chosen:", mode)
    for track in ("ms", "rgb"):
        tag = f"_{mode}"
        src = M / f"yield_{track}{tag}.joblib"
        if not src.exists():
            tag = "_absrel" if track == "rgb" else tag
            src = M / f"yield_{track}{tag}.joblib"
        if not src.exists():
            print("missing", src)
            continue
        shutil.copy2(src, M / f"yield_{track}.joblib")
        shutil.copy2(O / f"yield_results_{track}{tag}.json", O / f"yield_results_{track}.json")
        shutil.copy2(O / f"yield_predictions_{track}{tag}.csv", O / f"yield_predictions_{track}.csv")
        print(f"{track}: promoted {src.name}")
    (O / "feature_mode_decision.json").write_text(json.dumps({"p1_best_rmse": scores, "chosen": mode}, indent=2))


if __name__ == "__main__":
    main()
