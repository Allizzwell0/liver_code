#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast auto-tune tumor post-processing hyper-parameters on validation split.
- RAM cache (gt/prob/liver)
- tqdm progress bar
- presets: fast/full/custom
- de-duplicate tubular-disabled configs
- slow-trial warning
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import blosc2
import numpy as np
from tqdm.auto import tqdm

from tumor_postprocess import TumorPostprocessConfig, postprocess_tumor_prob


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_int_list(s: str) -> List[int]:
    return [int(float(x)) for x in s.split(",") if x.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--prob_dir", type=str, required=True, help="dir containing <case>_tumor_prob.npy")
    p.add_argument("--liver_dir", type=str, default=None, help="optional: dir containing <case>_pred_liver.npy")
    p.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cases", type=int, default=-1)

    # metric
    p.add_argument("--metric", type=str, default="weighted", choices=["dice_present", "weighted"])
    p.add_argument("--alpha", type=float, default=0.7, help="weighted metric: alpha*present + (1-alpha)*absent_score")
    p.add_argument("--fp_scale", type=float, default=2000.0, help="absent_score=exp(-pred_vox/fp_scale)")

    # speed / search presets
    p.add_argument("--preset", type=str, default="fast", choices=["fast", "full", "custom"],
                   help="fast: fixed-thr only, no tubular; full: original exhaustive; custom: use list args + toggles")
    p.add_argument("--disable_hysteresis", action="store_true")
    p.add_argument("--disable_tubular", action="store_true", help="force only tubular_aspect=0,tubular_thickness=0")
    p.add_argument("--warn_trial_sec", type=float, default=30.0, help="warn if a single config takes too long")

    # custom lists (also used to override presets when preset=custom)
    p.add_argument("--thr_list", type=str, default="0.25,0.3,0.35,0.4,0.45,0.5")
    p.add_argument("--min_cc_list", type=str, default="0,20,50,100")
    p.add_argument("--tubular_aspect_list", type=str, default="0,8,10,12")
    p.add_argument("--tubular_thickness_list", type=str, default="0,3,5,8")

    p.add_argument("--q_high_list", type=str, default="99.0,99.5,99.7")
    p.add_argument("--seed_floor_list", type=str, default="0.4,0.5,0.6")
    p.add_argument("--low_ratio_list", type=str, default="0.4,0.5,0.6")
    p.add_argument("--low_floor_list", type=str, default="0.1,0.15,0.2")

    p.add_argument("--out_json", type=str, default="best_tumor_postprocess.json")
    p.add_argument("--disable_tqdm", action="store_true")
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
            if m.shape != shape_zyx:
                Z, Y, X = shape_zyx
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


def _preset_lists(args):
    if args.preset == "full":
        return {
            "thr_list": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            "min_cc_list": [0, 20, 50, 100],
            "tasp_list": [0.0, 8.0, 10.0, 12.0],
            "tthk_list": [0, 3, 5, 8],
            "q_high_list": [99.0, 99.5, 99.7],
            "seed_floor_list": [0.4, 0.5, 0.6],
            "low_ratio_list": [0.4, 0.5, 0.6],
            "low_floor_list": [0.1, 0.15, 0.2],
            "disable_hysteresis": False,
            "disable_tubular": False,
        }

    if args.preset == "fast":
        # 先快速拿一个好参数：固定阈值 + 无tubular + 无hysteresis
        return {
            "thr_list": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            "min_cc_list": [0, 20, 50, 100, 150, 200],
            "tasp_list": [0.0],
            "tthk_list": [0],
            "q_high_list": [99.5],
            "seed_floor_list": [0.5],
            "low_ratio_list": [0.5],
            "low_floor_list": [0.15],
            "disable_hysteresis": True,
            "disable_tubular": True,
        }

    # custom
    return {
        "thr_list": _parse_float_list(args.thr_list),
        "min_cc_list": _parse_int_list(args.min_cc_list),
        "tasp_list": _parse_float_list(args.tubular_aspect_list),
        "tthk_list": _parse_int_list(args.tubular_thickness_list),
        "q_high_list": _parse_float_list(args.q_high_list),
        "seed_floor_list": _parse_float_list(args.seed_floor_list),
        "low_ratio_list": _parse_float_list(args.low_ratio_list),
        "low_floor_list": _parse_float_list(args.low_floor_list),
        "disable_hysteresis": args.disable_hysteresis,
        "disable_tubular": args.disable_tubular,
    }


def _build_tubular_pairs(tasp_list: List[float], tthk_list: List[int], disable_tubular: bool):
    if disable_tubular:
        return [(0.0, 0)]

    # 归一化“关闭tubular”的冗余组合：asp<=0 或 thk<=0 都视为关闭
    pairs = []
    seen = set()
    for asp in tasp_list:
        for thk in tthk_list:
            if asp <= 0 or thk <= 0:
                key = (0.0, 0)
            else:
                key = (float(asp), int(thk))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


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

    # -------- preload cache (FAST) --------
    data_cache = []
    for cid in case_ids:
        seg = load_seg(preproc_dir, cid)
        gt = (seg == 2).astype(np.uint8)
        gt_sum = int(gt.sum())

        prob = load_prob(prob_dir, cid)
        if prob.shape != gt.shape:
            Z, Y, X = gt.shape
            prob = prob[:Z, :Y, :X]
        prob = prob.astype(np.float16, copy=False)  # 缓存用fp16省内存

        liver = load_liver_mask(liver_dir, cid, gt.shape)
        if liver is not None:
            liver = liver.astype(np.uint8, copy=False)

        data_cache.append((cid, gt, gt_sum, prob, liver))
    print(f"[Tune] cached {len(data_cache)} cases in RAM")

    # preset / search space
    cfg_lists = _preset_lists(args)
    if args.preset != "custom":
        # 显式flag优先
        if args.disable_hysteresis:
            cfg_lists["disable_hysteresis"] = True
        if args.disable_tubular:
            cfg_lists["disable_tubular"] = True

    thr_list = cfg_lists["thr_list"]
    min_cc_list = cfg_lists["min_cc_list"]
    q_high_list = cfg_lists["q_high_list"]
    seed_floor_list = cfg_lists["seed_floor_list"]
    low_ratio_list = cfg_lists["low_ratio_list"]
    low_floor_list = cfg_lists["low_floor_list"]
    tubular_pairs = _build_tubular_pairs(
        cfg_lists["tasp_list"], cfg_lists["tthk_list"], cfg_lists["disable_tubular"]
    )
    disable_hysteresis = cfg_lists["disable_hysteresis"]

    fixed_trials = len(thr_list) * len(min_cc_list) * len(tubular_pairs)
    hyst_trials = 0 if disable_hysteresis else (
        len(q_high_list) * len(seed_floor_list) * len(low_ratio_list) * len(low_floor_list)
        * len(min_cc_list) * len(tubular_pairs)
    )
    total_trials = fixed_trials + hyst_trials

    print(f"[Tune] preset={args.preset} tubular_pairs={tubular_pairs}")
    print(f"[Tune] total_trials={total_trials} total_case_evals≈{total_trials * len(case_ids)}")

    best = None

    def eval_cfg(cfg: TumorPostprocessConfig) -> float:
        present_scores = []
        absent_scores_list = []
        for cid, gt, gt_sum, prob16, liver in data_cache:
            prob = prob16.astype(np.float32, copy=False)
            pred = postprocess_tumor_prob(prob, liver, cfg)

            if gt_sum > 0:
                present_scores.append(dice_empty_aware(pred, gt))
            else:
                absent_scores_list.append(absent_score(pred, args.fp_scale))

        mean_present = float(np.mean(present_scores)) if present_scores else 0.0
        mean_absent = float(np.mean(absent_scores_list)) if absent_scores_list else 0.0
        if args.metric == "dice_present":
            return mean_present
        return float(args.alpha * mean_present + (1.0 - args.alpha) * mean_absent)

    pbar = tqdm(total=total_trials, desc="[Tune]", dynamic_ncols=True, disable=args.disable_tqdm)
    trials = 0
    try:
        # 1) fixed-threshold mode
        for thr in thr_list:
            for min_cc in min_cc_list:
                for asp, thk in tubular_pairs:
                    cfg = TumorPostprocessConfig(
                        thr=float(thr),
                        use_hysteresis=False,
                        min_cc=int(min_cc),
                        tubular_aspect=float(asp),
                        tubular_thickness=int(thk),
                    )

                    t0 = time.perf_counter()
                    score = eval_cfg(cfg)
                    dt = time.perf_counter() - t0
                    trials += 1

                    if (best is None) or (score > best["score"]):
                        best = {"score": score, "cfg": cfg.to_dict(), "mode": "fixed_thr"}

                    pbar.update(1)
                    pbar.set_postfix(
                        mode="fixed",
                        best=f"{best['score']:.4f}",
                        cur=f"{score:.4f}",
                        sec=f"{dt:.1f}",
                        refresh=False,
                    )
                    if dt >= args.warn_trial_sec:
                        print(
                            f"\n[Warn] slow trial {trials}/{total_trials}: "
                            f"fixed thr={thr} min_cc={min_cc} asp={asp} thk={thk} took {dt:.1f}s"
                        )

        # 2) hysteresis mode
        if not disable_hysteresis:
            for q in q_high_list:
                for seed_floor in seed_floor_list:
                    for low_ratio in low_ratio_list:
                        for low_floor in low_floor_list:
                            for min_cc in min_cc_list:
                                for asp, thk in tubular_pairs:
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

                                    t0 = time.perf_counter()
                                    score = eval_cfg(cfg)
                                    dt = time.perf_counter() - t0
                                    trials += 1

                                    if (best is None) or (score > best["score"]):
                                        best = {"score": score, "cfg": cfg.to_dict(), "mode": "hysteresis"}

                                    pbar.update(1)
                                    pbar.set_postfix(
                                        mode="hyst",
                                        best=f"{best['score']:.4f}",
                                        cur=f"{score:.4f}",
                                        sec=f"{dt:.1f}",
                                        refresh=False,
                                    )
                                    if dt >= args.warn_trial_sec:
                                        print(
                                            f"\n[Warn] slow trial {trials}/{total_trials}: "
                                            f"hyst q={q} seed={seed_floor} low_ratio={low_ratio} low_floor={low_floor} "
                                            f"min_cc={min_cc} asp={asp} thk={thk} took {dt:.1f}s"
                                        )
    finally:
        pbar.close()

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