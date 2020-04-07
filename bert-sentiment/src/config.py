import transformers
import os

MAX_LEN = 150
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 12

BERT_PATH = "bert-base-multilingual-cased"
# BERT_PATH = "/home/TILDE.LV/gaurish.thakkar/experiments/bert-twitter-fine-tunning/LatvianTwittermBERT-v1"

DATASET_LOCATION = "/home/TILDE.LV/gaurish.thakkar/experiments/bert-twitter-fine-tunning/data/"

MODEL_PATH = "model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=False
)
