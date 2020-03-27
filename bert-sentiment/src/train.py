import random
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
from utils import categorical_accuracy, label_encoder
from torch.utils.tensorboard import SummaryWriter


# SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

writer = SummaryWriter()
logger.add("experiment.log")


def run():

    train_file = config.DATASET_LOCATION + "gold.prep.train.csv"
    df_train = pd.read_csv(train_file).fillna("none")
    df_train.label = df_train.label.apply(label_encoder)

    valid_file = config.DATASET_LOCATION + "gold.prep.dev.csv"
    df_valid = pd.read_csv(valid_file).fillna("none")
    df_valid.label = df_valid.label.apply(label_encoder)

    test_file = config.DATASET_LOCATION + "eval.prep.test.csv"
    df_test = pd.read_csv(test_file).fillna("none")
    df_test.label = df_test.label.apply(label_encoder)
    
    logger.info(f"Train file: {train_file}")
    logger.info(f"Valid file: {valid_file}")
    logger.info(f"Test file: {test_file}")

    logger.info(f"Train size : {len(df_train):.4f}")
    logger.info(f"Valid size : {len(df_valid):.4f}")
    logger.info(f"Test size : {len(df_test):.4f}")

    train_dataset = dataset.BERTDataset(
        review=df_train.text.values,
        target=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4, shuffle=True
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

    test_dataset = dataset.BERTDataset(
        review=df_test.text.values,
        target=df_test.label.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        logger.info(f"epoch={epoch}")

        train_loss, train_acc = engine.train_fn(
            train_data_loader, model, optimizer, device, scheduler)

        for tag, parm in model.named_parameters():
            if parm.grad is not None:
                writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)

        outputs, targets, val_loss, val_acc = engine.eval_fn(
            valid_data_loader, model, device)
        val_mcc = metrics.matthews_corrcoef(outputs, targets)
        logger.info(f"val_MCC_Score = {val_mcc:.3f}")

        outputs, targets, test_loss, test_acc = engine.eval_fn(
            test_data_loader, model, device)
        test_mcc = metrics.matthews_corrcoef(outputs, targets)
        logger.info(f"test_MCC_Score = {test_mcc:.3f}")

        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
        writer.add_scalar('loss/train', train_loss, epoch) # data grouping by `slash`
        writer.add_scalar('loss/val', val_loss, epoch) # data grouping by `slash`
        writer.add_scalar('loss/test', test_loss, epoch) # data grouping by `slash`
        
        logger.info(
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}")
        writer.add_scalar('acc/train', train_acc, epoch) # data grouping by `slash`
        writer.add_scalar('acc/val', val_acc, epoch) # data grouping by `slash`
        writer.add_scalar('acc/test', test_acc, epoch) # data grouping by `slash`
        
        logger.info(f"val_mcc={val_acc:.3f}, test_mcc={test_acc:.3f}")
        writer.add_scalar('mcc/val', val_mcc, epoch) # data grouping by `slash`
        writer.add_scalar('mcc/test', test_mcc, epoch) # data grouping by `slash`

        accuracy = metrics.accuracy_score(targets, outputs)
        logger.info(f"Accuracy Score = {accuracy:.3f}")

        if accuracy > best_accuracy:
            print(f"Saving model with Accuracy Score = {accuracy:.3f}")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
