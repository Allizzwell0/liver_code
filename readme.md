# liver相关
这部分医学分割任务的相关命令

使用nnUNet需要设置环境变量，暂时未写进source，每次开终端手动设置
```bash
export nnUNet_raw="/home/my/data/liver_data/nnUNet_data/raw"
export nnUNet_preprocessed="/home/my/data/liver_data/nnUNet_data/preprocessed"
export nnUNet_results="/home/my/data/liver_data/nnUNet_data/results"
```
检查：
```bash
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
```

对MSD数据进行预处理，直接调用nnUNet标准库
```bash
nnUNetv2_convert_MSD_dataset \
    -i /home/my/data/liver_data/MSD/Task03_Liver

# fingerprint
nnUNetv2_extract_fingerprint -d 3

# plan
nnUNetv2_plan_experiment -d 3 -c 3d_fullres

#prepocess
nnUNetv2_preprocess -d 3 -c 3d_fullres --verify_dataset_integrity

```

写入预处理后数据路径
```bash
PREPROC_DIR=//home/my/data/liver_data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres
```

换卡了这里不用管了，但第二张比较空
由于cuda0正在WM，故需要将训练放在另一张显卡上：
```bash
export CUDA_VISIBLE_DEVICES=1
```

训练脚本为src下的train.py
可以终止，从last/best继续训练
分为两阶段进行，先是liver，然后tumor
```bash
# liver 最后一项是中断后可以重新加载权重
python train.py \
  --preproc_dir $PREPROC_DIR \
  --stage liver \
  --epochs 1000 \
  --batch_size 2 \
  --patch_size 128 128 128 \
  --save_dir train_logs/lits_3dfullres_like_fix \
  --resume train_logs/lits_3dfullres_like_fix/liver_interrupt.pth 

# tumor 
python train.py \
  --preproc_dir $PREPROC_DIR \
  --stage tumor \
  --epochs 1000 \
  --batch_size 2 \
  --patch_size 128 128 128 \
  --save_dir train_logs/lits_3dfullres_like_tumor \
  --resume train_logs/lits_3dfullres_like_tumor/tumor_best.pth
```

训练时的eval是随机patch计算的，最后对模型进行完整的eval：
```bash
python eval_liver_full.py \
  --preproc_dir $PREPROC_DIR \
  --ckpt train_logs/lits_3dfullres_like_fix/liver_best.pth \
  --out_dir eval_liver_full_val \
  --split val \
  --patch_size 128 128 128 \
  --stride 64 64 64 \
  --save_npy
```

得到完整模型参数后，会进行两段式infer，输入为处理后的数据，输出为liver_mask, tumor_mask, 3class_mask
```bash
# 单个推理
python infer.py \
  --preproc_dir $PREPROC_DIR \
  --ckpt_liver /home/my/liver/liver_code/src/train_logs/lits_3dfullres_like/liver_best.pth \
  --ckpt_tumor /home/my/liver/liver_code/src/train_logs/lits_3dfullres_like_tumor_2/tumor_best.pth \
  --out_dir /home/my/data/liver_data/eval_outputs/LiTS_data \
  --case_id Dataset003_Liver_0000

# 全部测试
python infer.py \
  --preproc_dir $PREPROC_DIR \
  --ckpt_liver /home/my/liver/liver_code/src/train_logs/lits_3dfullres_like/liver_best.pth \
  --ckpt_tumor /home/my/liver/liver_code/src/train_logs/lits_3dfullres_like_tumor_2/tumor_best.pth \
  --out_dir /home/my/data/liver_data/eval_outputs/LiTS_data
```

查看训练曲线：
```bash
tensorboard --logdir train_logs --port 6006
```

后续想在模型基础上对自己的数据（CT .nrrd）进行一些测试，此处记录数据预处理及输入方式
预处理代码保存在dataload下，使用方法：
发现存在不同医学图像方向不同的情况，比如现在使用的LiTS数据集是RAS，对预处理部分增加方向选项
```bash
# 无标注数据
python dataload/preprocess_ct_data.py \
  --input_nrrd /home/my/data/liver_data/self_data/RAS_CT/6_RAS.nrrd \
  --plans_json /home/my/data/liver_data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans.json \
  --case_id my_case001 \
  --out_dir /home/my/data/liver_data/self_data/prepocessed_data \

# 标注数据
python dataload/preprocess_ct_data.py \
  --input_nrrd /home/my/data/liver_data/self_data/RAS_CT/6_RAS.nrrd \
  --seg_nrrd /path/to/seg.nrrd \
  --plans_json /home/my/data/liver_data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans.json \
  --case_id my_case001 \
  --out_dir /home/my/data/liver_data/self_data/prepocessed_data \
```
同时增加新的脚本，对nrrd转换坐标
```bash
python dataload/reorient_to_RAS.py \
    --input /home/my/data/liver_data/self_data/origin_CT \
    --output /home/my/data/liver_data/self_data/RAS_CT
```

完成上述处理后，在推理前重新设置PREPROC_DIR
```bash
PREPROC_DIR=//home/my/data/liver_data/self_data/prepocessed_data
```

之后infer命令可使用

发现重采样处理后原CT和seg并不能对应，后续进行debug，现在直接在本地slicer查看验证，此外生成对应的gif，方便查看
```bash
  python make_seg_gif.py \
    --img_b2nd /home/my/data/liver_data/self_data/prepocessed_data/my_case001.b2nd \
    --seg_npy  /home/my/data/liver_data/eval_outputs/my_data/my_case001_pred_3class.npy \
    --out_gif  /home/my/data/liver_data/eval_outputs/gif/my_case_001_z.gif \
    --axis z --fps 8
```

为了3Dslicer可以查看，将数据和推理结果合成nrrd文件：
```bash
python export_nrrd.py \
    --orig_ct /home/my/data/liver_data/self_data/RAS_CT/6_RAS.nrrd \
    --seg_npy /home/my/data/liver_data/eval_outputs/my_data/my_case001_pred_3class.npy \
    --spacing 1.0 0.767578125 0.767578125 \
    --out_seg /home/my/data/liver_data/self_data/slicer/my_case001_seg_on_orig.nrrd \
    --flip_z
```

记录新编号与原编号关系：
1-6 2-8 3-9 4-14

新加脚本用于对比liver分割效果
```bash

PREPROC_DIR=/home/my/data/liver_data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres
python compare.py \
  --preproc_dir $PREPROC_DIR \
  --ckpt train_logs/lits_3dfullres_like/liver_best.pth \
  --out_dir /home/my/data/liver_data/eval_outputs/compare \
  --split val \
  --patch_size 128 128 128 \
  --stride 64 64 64 \
    --seed 0 \
  --max_cases 10 \
  --run_totalseg 
```
