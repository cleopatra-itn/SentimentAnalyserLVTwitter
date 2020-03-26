import transformers
import os

MAX_LEN = 150
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH="bert-base-multilingual-cased"
# BERT_PATH = "/media/gaurish/angela/projects/lv-twitter-sa/bert-lm/models/LatvianTwitterBERTo-small-v1/"
# BERT_PATH = "/media/gaurish/angela/projects/lv-twitter-sa/bert-twitter-language-pretraining/models/LatvianTwittermBERT-v1"
DATASET_LOCATION="/media/gaurish/angela/projects/lv-twitter-sa/lv-twitter-data/"
# TOKENIZER_PATH = "/media/gaurish/angela/projects/lv-twitter-sa/bert-lm/models/lvtwitterberto/bpe/"
MODEL_PATH = "model.bin"
# TRAINING_FILE = "../input/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    # "bert-base-multilingual-cased",
    BERT_PATH,
    do_lower_case=False
)
