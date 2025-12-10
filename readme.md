# liver相关
这部分医学分割任务的相关命令

使用nnUNet需要设置环境变量，暂时未写进source，每次开终端手动设置
```bash
export nnUNet_raw="/home/mayue/WorldModel/Dreamer/liver/data/nnUNet_data/raw"
export nnUNet_preprocessed="/home/mayue/WorldModel/Dreamer/liver/data/nnUNet_data/preprocessed"
export nnUNet_results="/home/mayue/WorldModel/Dreamer/liver/data/nnUNet_data/results"
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
    -i /home/mayue/WorldModel/Dreamer/liver/data/MSD/Task03_Liver
```

写入预处理后数据路径
```bash
PREPROC_DIR=/home/mayue/WorldModel/Dreamer/liver/data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres
```

由于cuda0正在WM，故需要将训练放在另一张显卡上：
```bash
export CUDA_VISIBLE_DEVICES=1
```

训练脚本为src下的train.py
```bash
python train.py   --preproc_dir $PREPROC_DIR   --stage liver   --epochs 200   --batch_size 1   --patch_size 96 160 160   --save_dir train_logs/lits_b2nd
```