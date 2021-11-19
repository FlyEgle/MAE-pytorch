# #!/bin/bash
# OMP_NUM_THREADS=1
# MKL_NUM_THREADS=1
# export OMP_NUM_THREADS
# export MKL_NUM_THREADS
# cd /data/jiangmingchao/data/code/ImageClassification;
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_mae.py \
# --batch_size 256 \
# --num_workers 32 \
# --lr 1.5e-4 \
# --optimizer_name "adamw" \
# --cosine 1 \
# --max_epochs 300 \
# --warmup_epochs 40 \
# --num-classes 1000 \
# --crop_size 224 \
# --patch_size 16 \
# --color_prob 0.0 \
# --calculate_val 0 \
# --weight_decay 5e-2 \
# --lars 0 \
# --mixup 0.0 \
# --smoothing 0.0 \
# --train_file /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w.txt \
# --val_file /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt \
# --checkpoints-path /data/jiangmingchao/data/AICutDataset/VIT_MAE/vit_tiny_mae_300epoch_pretrain/checkpoints \
# --log-dir /data/jiangmingchao/data/AICutDataset/VIT_MAE/vit_tiny_mae_300epoch_pretrain/log_dir

#!/bin/bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
cd /data/jiangmingchao/data/code/ImageClassification;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_mae.py \
--batch_size 256 \
--num_workers 32 \
--lr 1e-3 \
--optimizer_name "adamw" \
--cosine 1 \
--max_epochs 300 \
--finetune 1 \
--warmup_epochs 20 \
--num-classes 1000 \
--crop_size 224 \
--patch_size 16 \
--color_prob 0.0 \
--calculate_val 1 \
--weight_decay 5e-2 \
--lars 0 \
--mixup 0.0 \
--cutmix 0.0 \
--smoothing 0.1 \
--train_file /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w.txt \
--val_file /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt \
--checkpoints-path /data/jiangmingchao/data/AICutDataset/VIT_MAE/vit_tiny_mae_300epoch_finetune/checkpoints \
--log-dir /data/jiangmingchao/data/AICutDataset/VIT_MAE/vit_tiny_mae_300epoch_finetune/log_dir