"""Step 3 - train the U-Net vegetation segmenter and evaluate it (Experiment 2).

Training data: PhenoBench UAV images with human-annotated semantic masks.
Classes 1-4 (crop, weed, partial crop, partial weed) are merged into
"vegetation" and class 0 (soil) is "background".

* PhenoBench train subset -> 90 % training / 10 % model-selection validation
* PhenoBench val subset   -> held-out TEST set (never used during training)

The same test images are also segmented with the classical ExG+Otsu baseline,
so the comparison is like-for-like.

Usage:  python scripts/03_train_segmentation.py --epochs 25
Output: models/unet_vegetation.pt, outputs/segmentation_metrics.json,
        outputs/segmentation_history.csv, docs/figures/exp2_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartyield import config  # noqa: E402
from smartyield.preprocessing import augment, normalize_for_cnn  # noqa: E402
from smartyield.segmentation.infer import exg_otsu_mask, seg_metrics  # noqa: E402
from smartyield.segmentation.unet import UNet, dice_bce_loss  # noqa: E402

LOAD_SIDE = 512   # PhenoBench 1024 px tiles are loaded at half resolution


def load_split(split: str) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    d = config.PHENOBENCH_DIR / split
    names = sorted(p.name for p in (d / "images").glob("*.png") if (d / "semantics" / p.name).exists())
    imgs, masks = [], []
    for n in names:
        img = cv2.cvtColor(cv2.imread(str(d / "images" / n)), cv2.COLOR_BGR2RGB)
        sem = cv2.imread(str(d / "semantics" / n), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (LOAD_SIDE, LOAD_SIDE), interpolation=cv2.INTER_AREA)
        sem = cv2.resize(sem, (LOAD_SIDE, LOAD_SIDE), interpolation=cv2.INTER_NEAREST)
        imgs.append(img)
        masks.append((sem > 0).astype(np.float32))
    return imgs, masks, names


def sample_batch(imgs, masks, bs, crop, rng):
    xs, ys = [], []
    for _ in range(bs):
        i = int(rng.integers(len(imgs)))
        img, m = imgs[i], masks[i]
        s = rng.uniform(0.35, 1.0)                       # scale augmentation (coarser GSD)
        side = int(LOAD_SIDE * s)
        img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (side, side), interpolation=cv2.INTER_NEAREST)
        if side < crop:
            pad = crop - side
            img = cv2.copyMakeBorder(img, 0, pad, 0, pad, cv2.BORDER_REFLECT)
            m = cv2.copyMakeBorder(m, 0, pad, 0, pad, cv2.BORDER_REFLECT)
            side = crop
        y0, x0 = rng.integers(0, side - crop + 1, size=2)
        img, m = img[y0:y0 + crop, x0:x0 + crop], m[y0:y0 + crop, x0:x0 + crop]
        img, m = augment(img, m, rng)
        xs.append(normalize_for_cnn(img))
        ys.append(m[None])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys).astype(np.float32))


@torch.no_grad()
def predict(model, img):
    x = torch.from_numpy(normalize_for_cnn(img))[None]
    return torch.sigmoid(model(x))[0, 0].numpy()


def evaluate(model, imgs, masks) -> dict:
    agg = np.zeros(4)
    per = []
    for img, m in zip(imgs, masks):
        p = predict(model, img) > 0.5
        r = seg_metrics(p, m > 0.5)
        agg += [r["tp"], r["fp"], r["fn"], r["tn"]]
        per.append(r)
    tp, fp, fn, tn = agg
    return {
        "iou": tp / (tp + fp + fn), "dice": 2 * tp / (2 * tp + fp + fn),
        "precision": tp / (tp + fp), "recall": tp / (tp + fn), "accuracy": (tp + tn) / agg.sum(),
        "mean_image_iou": float(np.mean([r["iou"] for r in per])),
        "std_image_iou": float(np.std([r["iou"] for r in per])),
        "n_images": len(imgs),
    }


def baseline(imgs, masks) -> dict:
    agg = np.zeros(4)
    per = []
    for img, m in zip(imgs, masks):
        r = seg_metrics(exg_otsu_mask(img), m > 0.5)
        agg += [r["tp"], r["fp"], r["fn"], r["tn"]]
        per.append(r)
    tp, fp, fn, tn = agg
    return {
        "iou": tp / (tp + fp + fn), "dice": 2 * tp / (2 * tp + fp + fn),
        "precision": tp / (tp + fp), "recall": tp / (tp + fn), "accuracy": (tp + tn) / agg.sum(),
        "mean_image_iou": float(np.mean([r["iou"] for r in per])),
        "std_image_iou": float(np.std([r["iou"] for r in per])),
        "n_images": len(imgs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40, help="batches per epoch")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=12, help="channels of the first U-Net level")
    a = ap.parse_args()

    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    tr_imgs, tr_masks, tr_names = load_split("train")
    te_imgs, te_masks, te_names = load_split("val")
    idx = rng.permutation(len(tr_imgs))
    n_val = max(1, len(idx) // 10)
    va = [int(i) for i in idx[:n_val]]
    tr = [int(i) for i in idx[n_val:]]
    va_imgs, va_masks = [tr_imgs[i] for i in va], [tr_masks[i] for i in va]
    tr_imgs, tr_masks = [tr_imgs[i] for i in tr], [tr_masks[i] for i in tr]
    print(f"train {len(tr_imgs)}  model-selection val {len(va_imgs)}  held-out test {len(te_imgs)}")

    widths = tuple(a.base * 2 ** i for i in range(5))
    model = UNet(widths=widths)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps)

    hist, best, best_state = [], -1.0, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        for _ in range(a.steps):
            x, y = sample_batch(tr_imgs, tr_masks, a.bs, a.crop, rng)
            loss = dice_bce_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        v = evaluate(model, va_imgs, va_masks)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_iou": v["iou"], "val_dice": v["dice"],
                     "elapsed_s": round(time.time() - t0, 1)})
        print(f"ep {ep:02d} loss {np.mean(losses):.4f} val IoU {v['iou']:.4f} Dice {v['dice']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v["iou"] > best:
            best = v["iou"]
            best_state = {k: t.clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    test_unet = evaluate(model, te_imgs, te_masks)
    test_exg = baseline(te_imgs, te_masks)
    torch.save({"state_dict": best_state, "widths": list(widths),
                "meta": {"trained_on": "PhenoBench v1.1 subset (UAV, human-annotated)", "inference_scale": 1.0,
                         "best_val_iou": best, "epochs": a.epochs}},
               config.MODELS_DIR / "unet_vegetation.pt")
    pd.DataFrame(hist).to_csv(config.OUTPUTS_DIR / "segmentation_history.csv", index=False)
    res = {"dataset": "PhenoBench v1.1 subset", "n_train": len(tr_imgs), "n_model_selection_val": len(va_imgs),
           "test_split": "PhenoBench official val split (held out)",
           "test_unet": test_unet, "test_exg_otsu_baseline": test_exg, "best_val_iou": best,
           "train_seconds": round(time.time() - t0, 1)}
    (config.OUTPUTS_DIR / "segmentation_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # qualitative figure: image | ground truth | ExG+Otsu | U-Net
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = np.linspace(0, len(te_imgs) - 1, 4).astype(int)
    fig, ax = plt.subplots(len(pick), 4, figsize=(12, 3 * len(pick)))
    for r, i in enumerate(pick):
        p = predict(model, te_imgs[i]) > 0.5
        e = exg_otsu_mask(te_imgs[i])
        for c, (im, t) in enumerate([(te_imgs[i], te_names[i]), (te_masks[i], "ground truth"),
                                     (e, f"ExG+Otsu IoU {seg_metrics(e, te_masks[i] > .5)['iou']:.2f}"),
                                     (p, f"U-Net IoU {seg_metrics(p, te_masks[i] > .5)['iou']:.2f}")]):
            ax[r, c].imshow(im, cmap="gray")
            ax[r, c].set_title(t, fontsize=8)
            ax[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp2_segmentation_examples.png", dpi=110)
    h = pd.DataFrame(hist)
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(h.epoch, h.train_loss, label="train loss (BCE+Dice)")
    ax1.set_xlabel("epoch")
    ax2 = ax1.twinx()
    ax2.plot(h.epoch, h.val_iou, color="tab:green", label="val IoU")
    fig.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "exp2_training_curve.png", dpi=110)


if __name__ == "__main__":
    main()
