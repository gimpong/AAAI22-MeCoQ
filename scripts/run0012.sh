#!/bin/bash
cd ..

python main.py \
    --notes CifarI64bitsSymm \
    --device cuda:$1 \
    --dataset CIFAR10 --protocal I \
    --trainable_layer_num 2 \
    --M 8 \
    --feat_dim 128 \
    --T 0.4 \
    --hp_beta 1e-3 \
    --hp_lambda 0.05 \
    --mode debias --pos_prior 0.1 \
    --queue_begin_epoch 3 \
    --symmetric_distance \
    --topK 1000



cd -
