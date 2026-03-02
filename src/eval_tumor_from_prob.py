import os
import argparse
from pathlib import Path

import numpy as np
import blosc2
from tqdm import tqdm

from tumor_postprocess import TumorPostprocessConfig, postprocess_tumor_prob


def list_case_ids(preproc_dir: Path, split: str, train_ratio: float, seed: int):
    case_ids = sorted([p.stem for p in preproc_dir.glob("*.pkl")])
    if split == "all":
        return case_ids
    import random
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    n_train = int(len(case_ids) * train_ratio)
    return case_ids[:n_train] if split == "train" else case_ids[n_train:]


def load_seg(preproc_dir: Path, case_id: str) -> np.ndarray:
    seg_file = preproc_dir / f"{case_id}_seg.b2nd"
    seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams={"nthreads": 1})
    seg = seg_b[:].astype(np.int16)
    if seg.ndim == 4:
        seg = seg[0]
    return seg


def load_prob(prob_dir: Path, case_id: str) -> np.ndarray:
    p = prob_dir / f"{case_id}_tumor_prob.npy"
    if not p.is_file():
        raise FileNotFoundError(f"Missing prob: {p}")
    return np.load(p).astype(np.float32)


def load_liver_mask(liver_dir: Path, case_id: str, shape_zyx):
    # 只需要一个肝脏限制mask即可：优先 pred_liver.npy
    p = liver_dir / f"{case_id}_pred_liver.npy"
    if p.is_file():
        m = (np.load(p) > 0).astype(np.uint8)
        Z, Y, X = shape_zyx
        return m[:Z, :Y, :X]
    return None


def dice_empty_aware(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-5) -> float:
    p = (pred01 > 0).astype(np.float32)
    g = (gt01 > 0).astype(np.float32)
    ps = float(p.sum())
    gs = float(g.sum())
    if gs == 0.0 and ps == 0.0:
        return 1.0
    if gs == 0.0 and ps > 0.0:
        return 0.0
    inter = float((p * g).sum())
    return float((2.0 * inter + eps) / (ps + gs + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", required=True)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--liver_dir", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "all"])
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--case_ids", default=None, help="comma-separated case ids")
    ap.add_argument("--postprocess_json", default=None)

    # allow override
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--min_cc", type=int, default=None)

    args = ap.parse_args()

    preproc_dir = Path(args.preproc_dir)
    prob_dir = Path(args.prob_dir)
    liver_dir = Path(args.liver_dir)

    if args.case_ids:
        case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    else:
        case_ids = list_case_ids(preproc_dir, args.split, args.train_ratio, args.seed)

    # load cfg
    if args.postprocess_json:
        import json
        cfgj = json.load(open(args.postprocess_json, "r", encoding="utf-8"))
        cfgd = cfgj["cfg"] if "cfg" in cfgj else cfgj
        cfg = TumorPostprocessConfig(**cfgd)
    else:
        cfg = TumorPostprocessConfig(thr=0.425, use_hysteresis=False, min_cc=50, tubular_aspect=0.0, tubular_thickness=0)

    # override some keys
    if args.thr is not None:
        cfg.thr = float(args.thr)
        cfg.use_hysteresis = False
    if args.min_cc is not None:
        cfg.min_cc = int(args.min_cc)

    print("[Prob-Eval] cfg:", cfg)

    rows = []
    for cid in tqdm(case_ids, desc="Prob-Eval"):
        seg = load_seg(preproc_dir, cid)
        gt = (seg == 2).astype(np.uint8)

        prob = load_prob(prob_dir, cid)
        Z, Y, X = gt.shape
        prob = prob[:Z, :Y, :X]

        liver = load_liver_mask(liver_dir, cid, gt.shape)
        pred = postprocess_tumor_prob(prob, liver, cfg).astype(np.uint8)

        d = dice_empty_aware(pred, gt)
        rows.append((cid, d, int(gt.sum()), int(pred.sum()), float(prob.max())))

    mean = float(np.mean([r[1] for r in rows])) if rows else 0.0
    std = float(np.std([r[1] for r in rows])) if rows else 0.0

    print("case\tDice\tgt_vox\tpred_vox\tprob_max")
    for r in rows:
        print(f"{r[0]}\t{r[1]:.6f}\t{r[2]}\t{r[3]}\t{r[4]:.4f}")
    print(f"\nMean\t{mean:.6f}\nStd\t{std:.6f}")


if __name__ == "__main__":
    main()