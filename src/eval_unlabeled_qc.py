#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC / sanity-check for unlabeled inference outputs.

Reads:
- pred_liver_dir: <case>_pred_liver.npy
- pred_tumor_dir: <case>_pred_tumor.npy (optional)

Outputs:
- prints summary
- optionally writes CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_liver_dir", type=str, required=True)
    p.add_argument("--pred_tumor_dir", type=str, default=None)
    p.add_argument("--out_csv", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    liver_dir = Path(args.pred_liver_dir)
    tumor_dir = Path(args.pred_tumor_dir) if args.pred_tumor_dir else None

    liver_files = sorted(liver_dir.glob("*_pred_liver.npy"))
    if not liver_files:
        liver_files = sorted(liver_dir.glob("*_liver_pred.npy"))
    if not liver_files:
        raise RuntimeError(f"No liver prediction files found in {liver_dir}")

    rows = []
    for p in liver_files:
        cid = p.name.replace("_pred_liver.npy","").replace("_liver_pred.npy","")
        liver = (np.load(p) > 0).astype(np.uint8)
        liver_vox = int(liver.sum())

        tumor_vox = None
        if tumor_dir:
            tp = tumor_dir / f"{cid}_pred_tumor.npy"
            if tp.is_file():
                tumor = (np.load(tp) > 0).astype(np.uint8)
                tumor_vox = int(tumor.sum())

        rows.append((cid, liver_vox, tumor_vox))

    # print
    lv = np.array([r[1] for r in rows], dtype=np.float64)
    print("=" * 60)
    print(f"[QC] cases={len(rows)}")
    print(f"  liver vox mean={lv.mean():.1f} std={lv.std():.1f} min={lv.min():.0f} max={lv.max():.0f}")
    if tumor_dir:
        tv = np.array([r[2] if r[2] is not None else 0 for r in rows], dtype=np.float64)
        print(f"  tumor vox mean={tv.mean():.1f} std={tv.std():.1f} min={tv.min():.0f} max={tv.max():.0f}")
    print("=" * 60)

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "liver_voxels", "tumor_voxels"])
            for r in rows:
                w.writerow(list(r))
        print(f"[QC] wrote: {outp}")


if __name__ == "__main__":
    main()
