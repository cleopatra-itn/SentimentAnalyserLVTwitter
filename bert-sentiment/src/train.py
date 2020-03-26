import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from loguru import logger
from utils import categorical_accuracy

logger.add("experiment.log")

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def run():
    label_vec = {"0": 0, "1": 1, "-1": 2}
    df_train = pd.read_csv(config.DATASET_LOCATION+"train.csv").fillna("none")
    df_train.label = df_train.label.apply(
        lambda x: label_vec[x.replace("__label__", "")]
    )
    df_valid = pd.read_csv(config.DATASET_LOCATION+"test.csv").fillna("none")
    df_valid.label = df_valid.label.apply(
        lambda x: label_vec[x.replace("__label__", "")]
    )
    logger.info(f"Train size".format(len(df_train)))
    logger.info(f"Test size".format(len(df_valid)))

    train_dataset = dataset.BERTDataset(
        review=df_train.text.values,
        target=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.text.values,
        target=df_valid.label.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        train_loss, train_acc= engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        logger.info(f"epoch={epoch}")
        logger.info(f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")
        # logger.info(
        #     f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"
        # )
        # logger.info(
        #     f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"
        # )
        outputs, targets, val_loss, val_acc = engine.eval_fn(valid_data_loader, model, device)
        logger.info(f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
        
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


    
if __name__ == "__main__":
    run()