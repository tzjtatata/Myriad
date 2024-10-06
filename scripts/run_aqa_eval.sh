#! /bin/bash
export CUDA_VISIBLE_DEVICES=5
cd /home/lyz/vdb/codes/minigpt
python evaluation_aqa_dataset.py --cfg-path eval_configs/mvtec/noise_test/aqa_normal_cutpaste_ve\=aprilgan.yaml --task_type adroi