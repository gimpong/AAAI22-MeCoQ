#!/bin/bash
cd ..

python main.py \
    --notes Nuswide32bitsSymm \
    --device cuda:$1 \
    --dataset NUSWIDE \
    --trainable_layer_num 0 \
    --M 4 \
    --feat_dim 32 \
    --T 0.4 \
    --hp_beta 1e-2 \
    --hp_lambda 0.2 \
    --mode debias --pos_prior 0.15 \
    --queue_begin_epoch 10 \
    --topK 5000 \
    --symmetric_distance


cd -
