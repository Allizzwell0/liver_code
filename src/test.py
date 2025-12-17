import blosc2
import numpy as np
from pathlib import Path
import pickle as pkl

def inspect_case(b2nd_path, pkl_path, name):
    arr = blosc2.open(urlpath=str(b2nd_path), mode="r")[:].astype(np.float32)
    print(f"=== {name} ===")
    print("shape:", arr.shape)    # (C, Z, Y, X)
    print("min / max:", float(arr.min()), float(arr.max()))
    print("mean / std:", float(arr.mean()), float(arr.std()))

    with open(pkl_path, "rb") as f:
        props = pkl.load(f)
    print("spacing:", props.get("spacing", None))
    print("itk_origin:", props.get("itk_origin", None))
    print("itk_direction:", props.get("itk_direction", None))
    print()

root = Path("/home/my/data/liver_data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres")

# LiTS 里的一个训练病例
inspect_case(
    root / "liver_0.b2nd",
    root / "liver_0.pkl",
    "LiTS liver_0"
)

# 你自己的一个病例
self_root = Path("/home/my/data/liver_data/self_data/prepocessed_data")
inspect_case(
    self_root / "my_case001.b2nd",
    self_root / "my_case001.pkl",
    "My case 001"
)
