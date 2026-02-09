import os, numpy as np, blosc2
from pathlib import Path

PREPROC=os.environ.get("PREPROC_DIR") or "/home/my/data/liver_data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres"
PROB_DIR="/home/my/data/liver_data/eval_outputs/tumor_val"   # tumor_prob.npy所在
LIVER_DIR="/home/my/data/liver_data/train_logs/pred_liver_for_tumor"  # liver_pred.npy所在

cases=["liver_127","liver_87","liver_15","liver_39","liver_92","liver_73","liver_77"]
thr=0.3

def load_seg(case_id):
    seg_b=blosc2.open(urlpath=str(Path(PREPROC)/f"{case_id}_seg.b2nd"), mode="r", dparams={"nthreads":1})
    seg=seg_b[:]
    if seg.ndim==4: seg=seg[0]
    return seg

for cid in cases:
    p=np.load(os.path.join(PROB_DIR,f"{cid}_prob_tumor.npy"))  # (Z,Y,X)
    seg=load_seg(cid)
    gt=(seg==2)

    pred=(p>=thr)

    gt_vox=int(gt.sum())
    pred_vox=int(pred.sum())
    inter=int((gt & pred).sum())

    # 关键：GT区域上的概率统计
    if gt_vox>0:
        p_in=p[gt]
        max_in=float(p_in.max())
        mean_in=float(p_in.mean())
    else:
        max_in=mean_in=0.0

    # 与肝mask关系：GT tumor 是否落在你预测的 liver 内
    liver_path=os.path.join(LIVER_DIR,f"{cid}_pred_liver.npy")
    if os.path.isfile(liver_path):
        lv=np.load(liver_path)>0
        out_liver=int((gt & (~lv)).sum())
        frac_out = out_liver/max(1,gt_vox)
    else:
        out_liver=-1; frac_out=-1

    print(f"{cid}: gt={gt_vox} pred={pred_vox} inter={inter} "
          f"| prob_in_gt mean={mean_in:.4f} max={max_in:.4f} "
          f"| gt_outside_pred_liver={out_liver} ({frac_out:.2%})")
