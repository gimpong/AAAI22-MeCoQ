#!/bin/bash
cd ..

python main.py \
    --notes Flickr16bitsSymm \
    --device cuda:$1 \
    --dataset Flickr25K \
    --trainable_layer_num 0 \
    --M 2 \
    --feat_dim 32 \
    --T 0.4 \
    --hp_beta 1e-1 \
    --hp_lambda 0.5 \
    --mode debias --pos_prior 0.15 \
    --queue_begin_epoch 5 \
    --topK 5000 \
    --symmetric_distance


cd -
