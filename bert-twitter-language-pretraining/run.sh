#!/bin/bash

# Path that contains model.bin and config.json
MODELPATH=

# Path for saving checkpoints
SAVEPATH=

# Training and evaluation files
TRAIN=
EVAL=

python run_language_modeling.py \
    --model_name_or_path $MODELPATH \
    --model_type bert \
    --train_data_file $TRAIN \
    --eval_data_file $EVAL \
    --mlm \
    --output_dir $SAVEPATH \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 7 \
    --save_steps 1000 \
    --logging_steps 1000 \
    --save_total_limit 5 \
    --per_gpu_train_batch_size 4 \
    --evaluate_during_training \
    --seed 42 \
    --overwrite_output_dir