#! /bin/bash
cd /home/lyz/vdb/codes/minigpt
export OMP_NUM_THREADS=4
set -x
port=$(shuf -i25000-30000 -n1)
torchrun --nproc_per_node 2 --master_port $port train.py --cfg-path $1