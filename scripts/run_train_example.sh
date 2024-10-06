#! /bin/bash
cd /home/lyz/vdb/codes/minigpt
export CUDA_VISIBLE_DEVICES=0
python train.py --cfg-path train_configs/example.yaml --options run.iters_per_epoch=2000 run.max_epoch=4 
python train.py --cfg-path train_configs/example.yaml --options run.iters_per_epoch=2000 run.max_epoch=8 
python train.py --cfg-path train_configs/example.yaml --options run.iters_per_epoch=2000 run.max_epoch=12 