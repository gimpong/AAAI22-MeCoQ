#!/bin/bash
cd ..

python main.py \
    --notes Flickr32bits \
    --device cuda:$1 \
    --dataset Flickr25K \
    --trainable_layer_num 0 \
    --M 4 \
    --feat_dim 64 \
    --T 0.45 \
    --hp_beta 1e-1 \
    --hp_lambda 1 \
    --mode debias --pos_prior 0.15 \
    --queue_begin_epoch 5 \
    --topK 5000


cd -
