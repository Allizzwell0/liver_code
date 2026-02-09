#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-tune tumor post-processing hyper-parameters on a labeled validation split.

This script DOES NOT retrain the network. It searches postprocess parameters using saved tumor probability maps:
  <prob_dir>/<case>_tumor_prob.npy  (float32, Z,Y,X)

Optionally, provide a liver prediction directory to restrict tumor to liver region:
  <liver_dir>/<case>_pred_liver.npy (uint8, Z,Y,X)

It outputs a JSON config that you can feed to infer_tumor_full.py / eval_tumor_full.py.

Example:
  python tune_tumor_postprocess.py \
    --preproc_dir $PREPROC_DIR \
    --prob_dir train_logs/lits_tumor_attn/val_prob \
    --liver_dir train_logs/pred_liver_for_tumor \
    --split val --train_ratio 0.8 --seed 0 \
    --out_json best_tumor_postprocess.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import blosc2
import numpy as np

from tumor_postprocess import TumorPostprocessConfig, postprocess_tumor_prob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--prob_dir", type=str, required=True, help="dir containing <case>_tumor_prob.npy")
    p.add_argument("--liver_dir", type=str, default=None, help="optional: dir containing <case>_pred_liver.npy")
    p.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cases", type=int, default=-1)

    # metric: balance tumor-present dice and tumor-absent FP penalty
    p.add_argument("--metric", type=str, default="weighted", choices=["dice_present", "weighted"])
    p.add_argument("--alpha", type=float, default=0.7, help="weighted metric: alpha*present + (1-alpha)*absent_score")
    p.add_argument("--fp_scale", type=float, default=2000.0, help="absent_score=exp(-pred_vox/fp_scale)")

    p.add_argument("--out_json", type=str, default="best_tumor_postprocess.json")
    return p.parse_args()


def list_case_ids(preproc_dir: Path, split: str, train_ratio: float, seed: int) -> List[str]:
    case_ids = sorted([p.stem for p in preproc_dir.glob("*.pkl")])
    if split == "all":
        return case_ids
    import random
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    n_train = int(len(case_ids) * train_ratio)
    return case_ids[:n_train] if split == "train" else case_ids[n_train:]


def load_seg(preproc_dir: Path, case_id: str) -> np.ndarray:
    dparams = {"nthreads": 1}
    seg_file = preproc_dir / f"{case_id}_seg.b2nd"
    seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams=dparams)
    seg = seg_b[:].astype(np.int16)
    if seg.ndim == 4:
        seg = seg[0]
    return seg


def load_prob(prob_dir: Path, case_id: str) -> np.ndarray:
    for name in [f"{case_id}_tumor_prob.npy", f"{case_id}_prob_tumor.npy", f"{case_id}_pred_tumor_prob.npy"]:
        p = prob_dir / name
        if p.is_file():
            return np.load(p).astype(np.float32)
    raise FileNotFoundError(f"tumor prob not found for {case_id} in {prob_dir}")


def load_liver_mask(liver_dir: Optional[Path], case_id: str, shape_zyx: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if liver_dir is None:
        return None
    for name in [f"{case_id}_pred_liver.npy", f"{case_id}_liver_pred.npy"]:
        p = liver_dir / name
        if p.is_file():
            m = (np.load(p) > 0).astype(np.uint8)
            if m.shape == shape_zyx:
                return m
            # allow simple crop/pad mismatch: crop to min
            Z, Y, X = shape_zyx
            mz, my, mx = m.shape
            m = m[:Z, :Y, :X]
            if m.shape == shape_zyx:
                return m
    return None


def dice_empty_aware(pred01: np.ndarray, gt01: np.ndarray, smooth: float = 1e-5) -> float:
    p = (pred01 > 0).astype(np.float32)
    g = (gt01 > 0).astype(np.float32)
    ps = float(p.sum())
    gs = float(g.sum())
    if gs == 0.0 and ps == 0.0:
        return 1.0
    if gs == 0.0 and ps > 0.0:
        return 0.0
    inter = float((p * g).sum())
    return float((2.0 * inter + smooth) / (ps + gs + smooth))


def absent_score(pred01: np.ndarray, fp_scale: float) -> float:
    pv = float((pred01 > 0).sum())
    return float(np.exp(-pv / max(1.0, fp_scale)))


def main():
    args = parse_args()
    blosc2.set_nthreads(1)

    preproc_dir = Path(args.preproc_dir)
    prob_dir = Path(args.prob_dir)
    liver_dir = Path(args.liver_dir) if args.liver_dir else None

    case_ids = list_case_ids(preproc_dir, args.split, args.train_ratio, args.seed)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]
    print(f"[Tune] split={args.split} cases={len(case_ids)} prob_dir={prob_dir}")

    # Search space (kept small & practical)
    thr_list = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    min_cc_list = [0, 20, 50, 100]
    tubular_aspect_list = [0.0, 8.0, 10.0, 12.0]
    tubular_thickness_list = [0, 3, 5, 8]

    # hysteresis params (adaptive thresholds)
    q_high_list = [99.0, 99.5, 99.7]
    seed_floor_list = [0.4, 0.5, 0.6]
    low_ratio_list = [0.4, 0.5, 0.6]
    low_floor_list = [0.1, 0.15, 0.2]

    best = None

    def eval_cfg(cfg: TumorPostprocessConfig) -> float:
        present_scores = []
        absent_scores = []
        for cid in case_ids:
            seg = load_seg(preproc_dir, cid)
            gt = (seg == 2).astype(np.uint8)
            prob = load_prob(prob_dir, cid)
            if prob.shape != gt.shape:
                # crop to min (shouldn't happen if you saved prob from same preproc)
                Z, Y, X = gt.shape
                prob = prob[:Z, :Y, :X]
            liver = load_liver_mask(liver_dir, cid, gt.shape)

            pred = postprocess_tumor_prob(prob, liver, cfg)

            if gt.sum() > 0:
                present_scores.append(dice_empty_aware(pred, gt))
            else:
                absent_scores.append(absent_score(pred, args.fp_scale))

        mean_present = float(np.mean(present_scores)) if len(present_scores) else 0.0
        mean_absent = float(np.mean(absent_scores)) if len(absent_scores) else 0.0
        if args.metric == "dice_present":
            return mean_present
        return float(args.alpha * mean_present + (1.0 - args.alpha) * mean_absent)

    # grid search: (1) fixed-thr mode (2) adaptive hysteresis mode
    trials = 0
    for thr in thr_list:
        for min_cc in min_cc_list:
            for asp in tubular_aspect_list:
                for thk in tubular_thickness_list:
                    cfg = TumorPostprocessConfig(
                        thr=float(thr),
                        use_hysteresis=False,
                        min_cc=int(min_cc),
                        tubular_aspect=float(asp),
                        tubular_thickness=int(thk),
                    )
                    score = eval_cfg(cfg)
                    trials += 1
                    if (best is None) or (score > best["score"]):
                        best = {"score": score, "cfg": cfg.to_dict(), "mode": "fixed_thr"}

    for q in q_high_list:
        for seed_floor in seed_floor_list:
            for low_ratio in low_ratio_list:
                for low_floor in low_floor_list:
                    for min_cc in min_cc_list:
                        for asp in tubular_aspect_list:
                            for thk in tubular_thickness_list:
                                cfg = TumorPostprocessConfig(
                                    thr=-1.0,
                                    use_hysteresis=True,
                                    q_high=float(q),
                                    seed_floor=float(seed_floor),
                                    low_ratio=float(low_ratio),
                                    low_floor=float(low_floor),
                                    min_cc=int(min_cc),
                                    tubular_aspect=float(asp),
                                    tubular_thickness=int(thk),
                                )
                                score = eval_cfg(cfg)
                                trials += 1
                                if (best is None) or (score > best["score"]):
                                    best = {"score": score, "cfg": cfg.to_dict(), "mode": "hysteresis"}

    print(f"[Tune] trials={trials} best_score={best['score']:.6f} mode={best['mode']}")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = prob_dir / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"[Tune] saved: {out_json}")


if __name__ == "__main__":
    main()
