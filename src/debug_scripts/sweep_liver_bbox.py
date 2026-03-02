import argparse
import numpy as np
from scipy.ndimage import label

def largest_cc_bbox(mask):
    lab, n = label(mask.astype(np.uint8))
    if n == 0: return None, 0
    sizes = np.bincount(lab.ravel()); sizes[0]=0
    k = int(sizes.argmax())
    m = (lab==k)
    idx = np.where(m)
    z0,z1 = int(idx[0].min()), int(idx[0].max())
    y0,y1 = int(idx[1].min()), int(idx[1].max())
    x0,x1 = int(idx[2].min()), int(idx[2].max())
    return (z0,z1,y0,y1,x0,x1), int(m.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True)
    args = ap.parse_args()
    prob = np.load(args.prob).astype(np.float32)
    Z,Y,X = prob.shape

    print("prob stats:", prob.min(), prob.max(), prob.mean())
    for thr in [0.5,0.6,0.7,0.8,0.9,0.95,0.98]:
        m = (prob > thr)
        bb, vox = largest_cc_bbox(m)
        zr = None if bb is None else (bb[1]-bb[0]+1)
        print(f"thr={thr:>4}  vox={vox:>9}  zspan={zr}  bbox={bb}")

    print("\n[Complement check] (1-prob) as liver_prob candidate")
    c = 1.0 - prob
    for thr in [0.5,0.6,0.7,0.8,0.9]:
        m = (c > thr)
        bb, vox = largest_cc_bbox(m)
        zr = None if bb is None else (bb[1]-bb[0]+1)
        print(f"thr={thr:>4}  vox={vox:>9}  zspan={zr}  bbox={bb}")

if __name__ == "__main__":
    main()
