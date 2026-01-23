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
新加bbox采样以及更改坐标输入方式（归一化后输入网络）

改成三阶段，定位-精分-肿瘤
```bash
# liver 最后一项是中断后可以重新加载权重
python train.py \
  --preproc_dir $PREPROC_DIR \
  --stage liver \
  --save_dir train_logs/liver_coarse \
  --epochs 500 \
  --batch_size 2 \
  --lr 2e-4 \
  --num_workers 4 \
  --patch_size 128 128 128 \
  --train_ratio 0.8 \
  --seed 0 \
  --liver_use_bbox 0 \
  --resume train_logs/liver_coarse/liver_last.pth


# 第二阶段准备
python infer.py \
  --preproc_dir $PREPROC_DIR \
  --out_dir train_logs/liver_coarse/pred_liver \
  --ckpt_coarse train_logs/liver_coarse/liver_best.pth \
  --bbox_margin 24 \
  --save_prob

# 常规训练 GT
python train.py \
  --preproc_dir $PREPROC_DIR \
  --stage liver \
  --save_dir train_logs/liver_refine \
  --epochs 500 \
  --batch_size 2 \
  --lr 2e-4 \
  --num_workers 4 \
  --patch_size 128 128 128 \
  --train_ratio 0.8 \
  --seed 0 \
  --liver_use_bbox 1 \
  --liver_bbox_margin 16 \
  --resume train_logs/liver_refine/liver_last.pth


# 一阶段结果加入
python train.py \
  --preproc_dir $PREPROC_DIR \
  --stage liver \
  --save_dir train_logs/liver_refine \
  --epochs 1000 \
  --batch_size 2 \
  --lr 2e-4 \
  --num_workers 4 \
  --patch_size 128 128 128 \
  --train_ratio 0.8 \
  --seed 0 \
  --liver_use_bbox 1 \
  --liver_bbox_margin 16 \
  --liver_use_pred_bbox 1 \
  --liver_pred_dir train_logs/liver_coarse/pred_liver \
  --resume train_logs/liver_refine/liver_last.pth




# tumor之前使用自训网络进行liver裁剪
python infer.py \
  --preproc_dir $PREPROC_DIR \
  --out_dir train_logs/pred_liver_for_tumor \
  --ckpt_coarse train_logs/liver_coarse/liver_best.pth \
  --ckpt_refine train_logs/liver_refine/liver_best.pth \
  --bbox_margin 24 \
  --save_prob



# tumor 
python train.py \
  --preproc_dir $PREPROC_DIR \
  --stage tumor \
  --save_dir train_logs/lits_tumor_bbox \
  --epochs 1000 \
  --batch_size 2 \
  --lr 1e-3 \
  --num_workers 4 \
  --patch_size 96 160 160 \
  --train_ratio 0.8 \
  --seed 0 \
  --tumor_use_pred_liver 1 \
  --tumor_pred_liver_dir train_logs/pred_liver_for_tumor \
  --tumor_pred_bbox_ratio 0.5 \
  --tumor_add_liver_prior 1 \
  --tumor_prior_type prob \
  --tumor_bbox_margin 24 \
  --tumor_pos_ratio 0.6 \
  --tumor_hardneg_ratio 0.3 \
  --tumor_alpha 0.75 \
  --tumor_gamma 2.0 \
  --resume train_logs/lits_tumor_bbox/tumor_last.pth


```

训练时的eval是随机patch计算的，最后对模型进行完整的eval：
```bash
# 仅评估liver_coarse
python eval_liver_full.py \
  --preproc_dir $PREPROC_DIR \
  --ckpt train_logs/liver_coarse/liver_best.pth \
  --out_dir /home/my/data/liver_data/eval_outputs/liver_coarse_val \
  --split val --train_ratio 0.8 --seed 0 \
  --patch_size 128 128 128 --stride 64 64 64 \
  --thr 0.5

# 评估liver-cascade
python eval_liver_full.py \
  --preproc_dir $PREPROC_DIR \
  --ckpt_coarse train_logs/liver_coarse/liver_best.pth \
  --ckpt_refine train_logs/liver_refine/liver_best.pth \
  --out_dir /home/my/data/liver_data/eval_outputs/liver_cascade_val \
  --split val --train_ratio 0.8 --seed 0 \
  --patch_size 128 128 128 --stride 64 64 64 \
  --bbox_margin 24 \
  --thr 0.5

# 评估liver-refine上限
python eval_liver_full.py \
  --preproc_dir $PREPROC_DIR \
  --ckpt_coarse train_logs/liver_coarse/liver_best.pth \
  --ckpt_refine train_logs/liver_refine/liver_best.pth \
  --out_dir /home/my/data/liver_data/eval_outputs/liver_refine_gtbbox_val \
  --split val --train_ratio 0.8 --seed 0 \
  --patch_size 128 128 128 --stride 64 64 64 \
  --bbox_margin 24 \
  --use_gt_bbox \
  --thr 0.5

```

得到完整模型参数后，会进行两段式infer，输入为处理后的数据，输出为liver_mask, tumor_mask, 3class_mask
```bash
# 有标注时，先进行裁剪，在进行tumor分割
python infer.py \
  --preproc_dir $PREPROC_DIR \
  --out_dir /home/my/data/liver_data/eval_outputs/pred_liver_for_tumor_val \
  --ckpt_coarse train_logs/liver_coarse/liver_best.pth \
  --ckpt_refine train_logs/liver_refine/liver_best.pth \
  --bbox_margin 24 \
  --save_prob

python eval_tumor_full.py \
  --preproc_dir $PREPROC_DIR \
  --liver_dir /home/my/data/liver_data/eval_outputs/pred_liver_for_tumor_val \
  --ckpt_tumor train_logs/lits_tumor_bbox/tumor_best.pth \
  --out_dir /home/my/data/liver_data/eval_outputs/tumor_val \
  --split val --train_ratio 0.8 --seed 0 \
  --patch_size 96 160 160 --stride 48 80 80 \
  --bbox_margin 24 \
  --thr 0.5 \
  --min_cc 5

# GT数据测上限
python eval_tumor_full.py \
  --preproc_dir $PREPROC_DIR \
  --liver_dir /home/my/data/liver_data/eval_outputs/pred_liver_for_tumor_val \
  --ckpt_tumor train_logs/lits_tumor_bbox/tumor_best.pth \
  --out_dir /home/my/data/liver_data/eval_outputs/tumor_val_gtbbox \
  --split val --train_ratio 0.8 --seed 0 \
  --patch_size 96 160 160 --stride 48 80 80 \
  --bbox_margin 24 \
  --use_gt_liver_bbox \
  --thr 0.5 \
  --min_cc 20

# 无标注时过程类似
python infer.py \
  --preproc_dir $MY_PREPROC_DIR \
  --out_dir /home/my/data/liver_data/my_eval_outs/pred_liver \
  --ckpt_coarse train_logs/liver_coarse/liver_best.pth \
  --ckpt_refine train_logs/liver_refine/liver_best.pth \
  --bbox_margin 24 \
  --save_prob

python infer_tumor_full.py \
  --preproc_dir $MY_PREPROC_DIR \
  --out_dir /home/my/data/liver_data/my_eval_outs/pred_tumor \
  --ckpt_tumor train_logs/lits_tumor_bbox/tumor_best.pth \
  --liver_dir /home/my/data/liver_data/my_eval_outs/pred_liver \
  --patch_size 96 160 160 --stride 48 80 80 \
  --bbox_margin 24 \
  --thr 0.5 \
  --min_cc 20 \
  --save_prob \
  --save_seg

# 无label的分割体素统计
python eval_unlabeled_qc.py \
  --pred_liver_dir /home/my/data/liver_data/my_eval_outs/pred_liver \
  --pred_tumor_dir /home/my/data/liver_data/my_eval_outs/pred_tumor \
  --out_csv /home/my/data/liver_data/my_eval_outs/qc_summary.csv


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
    --seg_npy /home/my/data/liver_data/my_eval_outs/pred_tumor/my_case001_pred_seg.npy \
    --spacing 1.0 0.767578125 0.767578125 \
    --out_seg /home/my/data/liver_data/my_eval_outs/slicer/my_case001_seg_on_orig.nrrd \
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
