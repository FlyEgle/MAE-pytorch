# Masked Autoencoders Are Scalable Vision Learners

### 1. Introduction
This repo is the MAE-vit model which impelement with pytorch, no reference any reference code so this is a non-official version. Because of the limitation of time and machine, I only trained the vit-tiny model for encoder.
![mae](fig/MAE.png)

### 2. Enveriments
- python 3.7+
- pytorch 1.7.1 
- pillow
- timm  
- opencv-python

### 3. Model Config
- **Encoder**
    ```
    ```
- **Decoder**
    ```
    ```


### 3. Results
![decoder](fig/decoder.png)
I test the imagenet val images from our mae-vit-tiny

### 3. Training & Inference
- dataset prepare
    ```
    /data/home/imagenet/xxx.jpeg, 0
    /data/home/imagenet/xxx.jpeg, 1
    ...
    /data/home/imagenet/xxx.jpeg, 999
    ```
- training 
    1. Pretrain 
        ```bash
        #!/bin/bash
        OMP_NUM_THREADS=1
        MKL_NUM_THREADS=1
        export OMP_NUM_THREADS
        export MKL_NUM_THREADS
        cd /data/jiangmingchao/data/code/ImageClassification;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_mae.py \
        --batch_size 256 \
        --num_workers 32 \
        --lr 1.5e-4 \
        --optimizer_name "adamw" \
        --cosine 1 \
        --max_epochs 300 \
        --warmup_epochs 40 \
        --num-classes 1000 \
        --crop_size 224 \
        --patch_size 16 \
        --color_prob 0.0 \
        --calculate_val 0 \
        --weight_decay 5e-2 \
        --lars 0 \
        --mixup 0.0 \
        --smoothing 0.0 \
        --train_file /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w.txt \
        --val_file /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt \
        --checkpoints-path /data/jiangmingchao/data/AICutDataset/VIT_MAE/vit_tiny_mae_300epoch_pretrain/checkpoints \
        --log-dir /data/jiangmingchao/data/AICutDataset/VIT_MAE/vit_tiny_mae_300epoch_pretrain/log_dir
        ```
    
    2. Finetune
    
        model is training 





