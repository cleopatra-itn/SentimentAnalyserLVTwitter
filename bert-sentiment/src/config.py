import transformers
import os

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 12

DATASET_LOCATION = "/home/TILDE.LV/gaurish.thakkar/experiments/bert-twitter-fine-tunning/data/"
MODEL_PATH = "model.bin"

# BERT_PATH = "/home/TILDE.LV/gaurish.thakkar/experiments/bert-twitter-fine-tunning/LatvianTwittermBERT-v1"
BERT_PATH = "/home/TILDE.LV/gaurish.thakkar/experiments/bert-twitter-language-pretraining/models/LatvianTwittermBERT-v1"
# BERT_PATH = "bert-base-multilingual-cased"

# BertTokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=False
)

# Electra
# BERT_PATH = "/home/TILDE.LV/gaurish.thakkar/experiments/lmtuners/experiments/disc_lm_small/electra-small/discriminator/final"
#"/home/TILDE.LV/gaurish.thakkar/experiments/lmtuners/experiments/disc_lm_small/albert-small/final"

# TOKENIZER = transformers.BertTokenizer.from_pretrained(
#     "/home/TILDE.LV/gaurish.thakkar/experiments/lmtuners/experiments/disc_lm_small/bert-base-multilingual-cased-vocab.txt",
#     do_lower_case=False
# )

# ALBERT_CONFIG = transformers.AlbertConfig(
#         vocab_size=len(TOKENIZER), #.get_vocab_size(),
#         hidden_size=256,
#         embedding_size=128,
#         num_hidden_layers=12,
#         num_attention_heads=4,
#         intermediate_size=1024,
#         max_position_embeddings=128)