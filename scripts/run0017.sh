#!/bin/bash
cd ..

python main.py \
    --notes CifarII64bits \
    --device cuda:$1 \
    --dataset CIFAR10 --protocal II \
    --trainable_layer_num 2 \
    --M 8 \
    --feat_dim 96 \
    --T 0.35 \
    --hp_beta 1e-2 \
    --hp_lambda 0.05 \
    --mode debias --pos_prior 0.1 \
    --queue_begin_epoch 15 \
    --topK 1000


cd -
