import transformers
import os
import torch

MAX_LEN = 150 #256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5

# Processed training, development and evaluation files
TRAIN_PROC = ""
DEVEL_PROC = ""
EVAL_PROC = ""

# Path to save sentiment analysis model
MODEL_PATH = "model.bin"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to the best BERT model checkpoint adapted using bert-twitter-language-pretraining or just MBERT Raw Version
BERT_PATH = ""

# TODO check if lower casing is required
# BertTokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)

#####################################################################################################################################
# Electra
# Step 1: Model path
# BERT_PATH = "lmtuners/experiments/disc_lm_small/electra-small/discriminator/final"
# #"lmtuners/experiments/disc_lm_small/albert-small/final"

# # Step 2: Vocab and Lowercase setting
# TOKENIZER = transformers.BertTokenizer.from_pretrained(
# 	"lmtuners/experiments/disc_lm_small/lvtwitterbwpt-vocab-lower_accent.txt",
#     # "lmtuners/experiments/disc_lm_small/bert-base-multilingual-cased-vocab.txt",
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
