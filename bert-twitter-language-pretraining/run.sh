




python run_language_modeling.py \
    --model_name_or_path models/LatvianTwittermBERT-v1\
    --model_type bert \
    --config_name models/LatvianTwittermBERT-v1/config.json \
    --tokenizer_name bert-base-multilingual-cased \
    --train_data_file data/0-train.txt \
    --eval_data_file data/0-eval.txt \
    --mlm \
    --output_dir models/LatvianTwittermBERT-v2 \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 7 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --per_gpu_train_batch_size 8 \
    --evaluate_during_training \
    --seed 42 \
    --overwrite_output_dir \
