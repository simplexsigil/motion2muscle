#!/bin/bash

export MINT_DATA="/lsdf/data/activity/MuscleSim/musclesim_dataset"
export MOTION_DATA="/lsdf/users/dschneider-kf3609/workspace/HumanML3D/HumanML3D"
export BABEL_DATA="/lsdf/data/activity/BABEL/babel_v1-0_release"

export CUDA_VISIBLE_DEVICES=1

python3 train_vq_ms.py \
--batch-size 128 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 402 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output/vqvae \
--dataname mint \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name vqvae \
--print-iter 50 \
--eval-iter 1000 \
--window-size 28 \
--vq_dec_emb_width 402 \
--muscle_subset MUSINT_402