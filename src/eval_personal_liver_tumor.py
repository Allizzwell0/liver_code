#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import blosc2


def dice(pred01, gt01, eps=1e-5):
    p = (pred01 > 0).astype(np.float32)
    g = (gt01 > 0).astype(np.float32)
    ps, gs = float(p.sum()), float(g.sum())
    if gs == 0.0 and ps == 0.0:
        return 1.0
    if gs == 0.0 and ps > 0.0:
        return 0.0
    inter = float((p * g).sum())
    return float((2 * inter + eps) / (ps + gs + eps))


def load_seg_b2nd(preproc_dir: Path, case_id: str) -> np.ndarray:
    seg_path = preproc_dir / f"{case_id}_seg.b2nd"
    seg_b = blosc2.open(urlpath=str(seg_path), mode="r", dparams={"nthreads": 1})
    seg = seg_b[:].astype(np.int16)
    if seg.ndim == 4:
        seg = seg[0]
    return seg  # (Z,Y,X)


def crop_or_pad(arr: np.ndarray, shape_zyx):
    Z, Y, X = shape_zyx
    out = np.zeros(shape_zyx, dtype=arr.dtype)
    zz, yy, xx = arr.shape
    out[:min(Z, zz), :min(Y, yy), :min(X, xx)] = arr[:min(Z, zz), :min(Y, yy), :min(X, xx)]
    return out


def find_case_ids(pred_liver_dir: Path, pred_tumor_dir: Path):
    ids = set()
    if pred_liver_dir and pred_liver_dir.exists():
        for p in pred_liver_dir.glob("*_pred_liver.npy"):
            ids.add(p.name.replace("_pred_liver.npy", ""))
    if pred_tumor_dir and pred_tumor_dir.exists():
        for p in pred_tumor_dir.glob("*_tumor_pred.npy"):
            ids.add(p.name.replace("_tumor_pred.npy", ""))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liver_label", type=int, default=2, help="GT liver label value (personal data often 2)")
    ap.add_argument("--tumor_label", type=int, default=1, help="GT tumor label value (personal data often 1)")
    ap.add_argument("--preproc_dir", required=True, help="contains <case>_seg.b2nd for labeled cases")
    ap.add_argument("--pred_liver_dir", required=True, help="contains <case>_pred_liver.npy (mixed ok)")
    ap.add_argument("--pred_tumor_dir", required=True, help="contains <case>_tumor_pred.npy (mixed ok)")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--case_ids", default=None, help="comma-separated case ids (optional)")
    ap.add_argument("--require_both", action="store_true",
                    help="if set, only evaluate cases that have BOTH liver & tumor preds")
    args = ap.parse_args()

    preproc_dir = Path(args.preproc_dir)
    pred_liver_dir = Path(args.pred_liver_dir)
    pred_tumor_dir = Path(args.pred_tumor_dir)

    if args.case_ids:
        case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    else:
        case_ids = find_case_ids(pred_liver_dir, pred_tumor_dir)

    if not case_ids:
        raise RuntimeError("No predicted cases found in pred_liver_dir / pred_tumor_dir.")

    rows = []
    skipped_no_gt = []
    skipped_missing_pred = []

    for cid in case_ids:
        seg_path = preproc_dir / f"{cid}_seg.b2nd"
        if not seg_path.is_file():
            skipped_no_gt.append(cid)
            continue

        liver_pred_path = pred_liver_dir / f"{cid}_pred_liver.npy"
        tumor_pred_path = pred_tumor_dir / f"{cid}_tumor_pred.npy"

        has_liver_pred = liver_pred_path.is_file()
        has_tumor_pred = tumor_pred_path.is_file()

        if args.require_both and (not (has_liver_pred and has_tumor_pred)):
            skipped_missing_pred.append(cid)
            continue

        seg = load_seg_b2nd(preproc_dir, cid)
        gt_liver = (seg == args.liver_label).astype(np.uint8)
        gt_tumor = (seg == args.tumor_label).astype(np.uint8)

        liver_d = None
        tumor_d = None
        pred_liver_vox = None
        pred_tumor_vox = None

        if has_liver_pred:
            pl = np.load(liver_pred_path).astype(np.uint8)
            pl = crop_or_pad(pl, gt_liver.shape)
            liver_d = dice(pl, gt_liver)
            pred_liver_vox = int(pl.sum())

        if has_tumor_pred:
            pt = np.load(tumor_pred_path).astype(np.uint8)
            pt = crop_or_pad(pt, gt_tumor.shape)
            tumor_d = dice(pt, gt_tumor)
            pred_tumor_vox = int(pt.sum())

        rows.append((
            cid,
            liver_d if liver_d is not None else np.nan,
            tumor_d if tumor_d is not None else np.nan,
            int(gt_liver.sum()),
            int(gt_tumor.sum()),
            pred_liver_vox if pred_liver_vox is not None else -1,
            pred_tumor_vox if pred_tumor_vox is not None else -1,
        ))

    # print
    print("case\tLiverDice\tTumorDice\tgtL\tgtT\tpredL\tpredT")
    for r in rows:
        print(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}\t{r[3]}\t{r[4]}\t{r[5]}\t{r[6]}")

    liver_vals = [r[1] for r in rows if np.isfinite(r[1])]
    tumor_vals = [r[2] for r in rows if np.isfinite(r[2])]

    if liver_vals:
        print(f"\n[Liver] mean={np.mean(liver_vals):.6f} std={np.std(liver_vals):.6f} n={len(liver_vals)}")
    if tumor_vals:
        print(f"[Tumor] mean={np.mean(tumor_vals):.6f} std={np.std(tumor_vals):.6f} n={len(tumor_vals)}")

    print(f"\n[Skip] no GT seg: {len(skipped_no_gt)}")
    if skipped_no_gt:
        print("  examples:", ", ".join(skipped_no_gt[:20]))
    print(f"[Skip] missing preds (require_both): {len(skipped_missing_pred)}")
    if skipped_missing_pred:
        print("  examples:", ", ".join(skipped_missing_pred[:20]))

    # csv
    if args.out_csv:
        import csv
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case", "liver_dice", "tumor_dice", "gt_liver_vox", "gt_tumor_vox", "pred_liver_vox", "pred_tumor_vox"])
            for r in rows:
                w.writerow(r)
        print("saved:", out_csv)


if __name__ == "__main__":
    main()