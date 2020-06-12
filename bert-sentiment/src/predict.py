import datetime
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from absl import app, flags, logging
from loguru import logger
from sklearn import metrics, model_selection
from torch.utils.tensorboard import SummaryWriter

import config
import dataset
import engine
from model import BERTBaseUncased
from utils import categorical_accuracy, label_encoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()
logger.add("experiment.log")


def main(_):

    test_file = config.DATASET_LOCATION + "eval.prep.test.csv"
    df_test = pd.read_csv(test_file).fillna("none")
    # Commenting as there are no labels
    df_test.label = df_test.label.apply(label_encoder)
    
    logger.info(f"Bert Model: {config.BERT_PATH}")
    logger.info(f"Current date and time :{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Test size : {len(df_test):.4f}")

    test_dataset = dataset.BERTDataset(
        review=df_test.text.values,
        target=df_test.label.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=3
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTBaseUncased()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(device)))
    model.to(device)
  

    outputs, targets, test_loss, test_acc = engine.eval_fn(
        test_data_loader, model, device)
    test_mcc = metrics.matthews_corrcoef(outputs, targets)
    print(test_loss, test_acc)
    logger.info(f"test_MCC_Score = {test_mcc:.3f}")
       

if __name__ == "__main__":
    app.run(main)
