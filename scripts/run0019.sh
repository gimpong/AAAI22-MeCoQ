#!/bin/bash
cd ..

python main.py \
    --notes Nuswide16bits \
    --device cuda:$1 \
    --dataset NUSWIDE \
    --trainable_layer_num 0 \
    --M 2 \
    --feat_dim 64 \
    --T 0.2 \
    --hp_beta 1e-2 \
    --hp_lambda 1 \
    --mode debias --pos_prior 0.15 \
    --queue_begin_epoch 10 \
    --topK 5000


cd -
