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
python train.py   --preproc_dir $PREPROC_DIR   --stage liver   --epochs 200   --batch_size 2   --patch_size 96 160 160   --save_dir train_logs/lits_b2nd   --resume train_logs/lits_b2nd/liver_best.pth
```

查看训练曲线：
```bash
tensorboard --logdir train_logs --port 6006
```

后续想在模型基础上对自己的数据（CT .nrrd）进行一些测试，此处记录数据预处理及输入方式