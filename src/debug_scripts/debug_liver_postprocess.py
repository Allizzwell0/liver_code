import argparse
from pathlib import Path
import numpy as np
import blosc2

def try_import_scipy():
    try:
        from scipy.ndimage import label, binary_fill_holes
        return label, binary_fill_holes
    except Exception:
        return None, None

def load_b2nd(preproc_dir: Path, case_id: str) -> np.ndarray:
    data_file = preproc_dir / f"{case_id}.b2nd"
    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams={"nthreads": 1})
    arr = data_b[:].astype(np.float32)  # (C,Z,Y,X)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr

def bbox(mask01: np.ndarray):
    idx = np.where(mask01 > 0)
    if idx[0].size == 0:
        return None
    z0,z1 = int(idx[0].min()), int(idx[0].max())
    y0,y1 = int(idx[1].min()), int(idx[1].max())
    x0,x1 = int(idx[2].min()), int(idx[2].max())
    return (z0,z1,y0,y1,x0,x1)

def cc_stats(mask01: np.ndarray, prob: np.ndarray, topk: int = 5):
    label, _ = try_import_scipy()
    if label is None:
        return {"n": -1, "top": []}
    lab, n = label(mask01.astype(np.uint8), structure=np.ones((3,3,3), dtype=np.uint8))
    if n == 0:
        return {"n": 0, "top": []}
    flat_lab = lab.ravel()
    flat_prob = prob.ravel()
    size = np.bincount(flat_lab, minlength=n+1).astype(np.float64)
    sump = np.bincount(flat_lab, weights=flat_prob, minlength=n+1).astype(np.float64)
    meanp = sump / np.maximum(size, 1.0)
    rows = []
    for k in range(1, n+1):
        rows.append((k, int(size[k]), float(meanp[k]), float(sump[k])))
    rows.sort(key=lambda x: x[1], reverse=True)
    return {"n": n, "top": rows[:topk], "all_rows": rows, "lab": lab}

def keep_largest_cc(mask01: np.ndarray):
    st = cc_stats(mask01, np.zeros_like(mask01, dtype=np.float32), topk=1)
    if st["n"] <= 0:
        return mask01.astype(np.uint8)
    k = st["top"][0][0]
    label, _ = try_import_scipy()
    lab, _n = label(mask01.astype(np.uint8), structure=np.ones((3,3,3), dtype=np.uint8))
    return (lab == k).astype(np.uint8)

def fill_holes(mask01: np.ndarray):
    _, binary_fill_holes = try_import_scipy()
    if binary_fill_holes is None:
        return mask01.astype(np.uint8)
    return binary_fill_holes(mask01.astype(bool)).astype(np.uint8)

def keep_best_cc_by_meanprob(mask01: np.ndarray, prob: np.ndarray):
    st = cc_stats(mask01, prob, topk=10)
    if st["n"] <= 0:
        return mask01.astype(np.uint8)
    # 选 mean(prob) 最大的 cc（忽略背景0）
    best = max(st["all_rows"], key=lambda x: x[2])[0]
    label, _ = try_import_scipy()
    lab, _n = label(mask01.astype(np.uint8), structure=np.ones((3,3,3), dtype=np.uint8))
    return (lab == best).astype(np.uint8)

def fmt_mask(name, m, prob, img0):
    vox = int(m.sum())
    frac = vox / max(1, m.size)
    bb = bbox(m)
    st = cc_stats(m, prob, topk=5)
    air_thr = float(np.percentile(img0, 1.0))
    air = (img0 <= air_thr)
    air_frac = float((m.astype(bool) & air).sum()) / max(1, vox) if vox > 0 else 0.0
    print(f"\n[{name}] vox={vox} frac={frac:.4f} bbox={bb}  air_overlap={air_frac:.4f}  cc_n={st['n']}")
    if st["n"] > 0:
        print("  top cc by size: (id, size, meanProb, sumProb)")
        for r in st["top"]:
            print("   ", r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", required=True)
    ap.add_argument("--pred_liver_dir", required=True)
    ap.add_argument("--case_id", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--thr_bbox", type=float, default=0.8, help="for seed/bbox debug only")
    args = ap.parse_args()

    preproc_dir = Path(args.preproc_dir)
    pred_dir = Path(args.pred_liver_dir)
    cid = args.case_id

    prob_path = pred_dir / f"{cid}_prob_liver.npy"
    if not prob_path.is_file():
        raise FileNotFoundError(prob_path)

    img = load_b2nd(preproc_dir, cid)   # (C,Z,Y,X)
    img0 = img[0]                       # (Z,Y,X) normalized CT
    prob = np.load(prob_path).astype(np.float32)
    Z,Y,X = img0.shape
    prob = prob[:Z,:Y,:X]

    print("prob stats:", float(prob.min()), float(prob.max()), float(prob.mean()),
          "frac>0.1=", float((prob>0.1).mean()), "frac>0.5=", float((prob>0.5).mean()))
    print("scipy available:", try_import_scipy()[0] is not None)

    raw = (prob > float(args.thr)).astype(np.uint8)
    fmt_mask(f"RAW thr={args.thr}", raw, prob, img0)

    # your current infer.py postprocess: largest -> fill_holes -> largest
    m1 = keep_largest_cc(raw)
    fmt_mask("PP step1 keep_largest_cc", m1, prob, img0)
    m2 = fill_holes(m1)
    fmt_mask("PP step2 fill_holes", m2, prob, img0)
    m3 = keep_largest_cc(m2)
    fmt_mask("PP step3 keep_largest_cc", m3, prob, img0)

    # alternative: select cc by mean prob (often fixes 'body is largest cc')
    mb = keep_best_cc_by_meanprob(raw, prob)
    fmt_mask("ALT keep_best_cc_by_meanProb", mb, prob, img0)

    # bbox seed test
    seed = (prob > float(args.thr_bbox)).astype(np.uint8)
    fmt_mask(f"SEED thr_bbox={args.thr_bbox}", seed, prob, img0)

if __name__ == "__main__":
    main()
