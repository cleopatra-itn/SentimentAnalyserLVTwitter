import torch
import torch.nn as nn
from tqdm import tqdm
from utils import categorical_accuracy


def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        pred_labels = torch.argmax(outputs, dim=1)
        # (pred_labels == targets).sum().item()
        train_acc += categorical_accuracy(outputs, targets).item()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def eval_fn(data_loader, model, device):
    model.eval()
    eval_loss, eval_acc = 0.0, 0.0
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs, targets)
            eval_loss += loss.item()
            pred_labels = torch.argmax(outputs, axis=1)
            # (pred_labels == targets).sum().item()
            eval_acc += categorical_accuracy(outputs, targets).item()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.argmax(
                outputs, dim=1).cpu().detach().numpy().tolist())
    eval_loss /= len(data_loader)
    eval_acc /= len(data_loader)
    return fin_outputs, fin_targets, eval_loss, eval_acc
