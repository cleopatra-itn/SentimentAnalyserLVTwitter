import transformers
import os

MAX_LEN = 150 #256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5

# Folder to contain all the datasets
DATASET_LOCATION = "" # 
MODEL_PATH = "mbert-7epoch-gold-lower/model.bin"


# MBERT Raw Version
# BERT_PATH = "bert-base-multilingual-cased"

# 2 EPOCH Version
# BERT_PATH = "/home/TILDE.LV/gaurish.thakkar/experiments/bert-twitter-fine-tunning/LatvianTwittermBERT-v1"

# 7 EPOCH Version
BERT_PATH = "FFZG-cleopatra/bert-emoji-latvian-twitter"

# 7 EPOCH Version + emoticons
# BERT_PATH = "/home/TILDE.LV/gaurish.thakkar/experiments/bert-twitter-language-pretraining/models/LatvianTwittermBERT-v2/checkpoint-106000"

# TODO check if lower casing is required
# BertTokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)

#####################################################################################################################################
# Electra
# Step 1: Model path
# BERT_PATH = "/home/TILDE.LV/gaurish.thakkar/experiments/lmtuners/experiments/disc_lm_small/electra-small/discriminator/final"
# #"/home/TILDE.LV/gaurish.thakkar/experiments/lmtuners/experiments/disc_lm_small/albert-small/final"

# # Step 2: Vocab and Lowercase setting
# TOKENIZER = transformers.BertTokenizer.from_pretrained(
# 	"/home/TILDE.LV/gaurish.thakkar/experiments/lmtuners/experiments/disc_lm_small/lvtwitterbwpt-vocab-lower_accent.txt",
#     # "/home/TILDE.LV/gaurish.thakkar/experiments/lmtuners/experiments/disc_lm_small/bert-base-multilingual-cased-vocab.txt",
#     do_lower_case=True
# )

# ALBERT_CONFIG = transformers.AlbertConfig(
#         vocab_size=len(TOKENIZER), #.get_vocab_size(),
#         hidden_size=256,
#         embedding_size=128,
#         num_hidden_layers=12,
#         num_attention_heads=4,
#         intermediate_size=1024,
#         max_position_embeddings=128)
