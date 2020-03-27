import transformers
import os

MAX_LEN = 150
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5

BERT_PATH = "bert-base-multilingual-cased"
# BERT_PATH = "/media/gaurish/angela/projects/lv-twitter-sa/bert-twitter-language-pretraining/models/LatvianTwittermBERT-v1"

DATASET_LOCATION = "/media/gaurish/angela/projects/lv-twitter-sa/lv-twitter-data-csv/"

MODEL_PATH = "model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=False
)
