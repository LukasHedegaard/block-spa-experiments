#!/bin/bash

SERIALIZATION_DIR=seriealized_data
SQUAD_DATA=squad_data

# mkdir -p $SQUAD_DATA
# cd $SQUAD_DATA
# wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
# wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
# cd ..


# RANK=1 WORLD_SIZE=1 \
    # --local_rank 0 \
python block_movement_pruning/masked_run_squad.py \
    --output_dir $SERIALIZATION_DIR \
    --data_dir $SQUAD_DATA \
    --train_file train-v1.1.json \
    --predict_file dev-v1.1.json \
    --do_train --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path bert-base-uncased \
    --per_gpu_train_batch_size 16 \
    --warmup_steps 5400 \
    --num_train_epochs 10 \
    --learning_rate 3e-5 --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold 0.15 \
    --initial_warmup 1 --final_warmup 2 \
    --pruning_method topK \
    --mask_init constant \
    --mask_scale 0. \
