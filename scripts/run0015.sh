#!/bin/bash
cd ..

python main.py \
    --notes CifarII32bits \
    --device cuda:$1 \
    --dataset CIFAR10 --protocal II \
    --trainable_layer_num 2 \
    --M 4 \
    --feat_dim 48 \
    --T 0.35 \
    --hp_beta 5e-3 \
    --hp_lambda 0.1 \
    --mode debias --pos_prior 0.1 \
    --queue_begin_epoch 15 \
    --topK 1000


cd -
