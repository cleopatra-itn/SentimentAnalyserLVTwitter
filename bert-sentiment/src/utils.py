import torch
import config


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def label_encoder(x):
    label_vec = {"0": 0, "1": 1, "-1": 2}
    return label_vec[x.replace("__label__", "")]

def label_decoder(x):
    label_vec = { 0:"U",  1:"P",  2:"N"}
    return label_vec[x]

def write_embeddings_to_file(model, x):
    pass
    # write the
    # write it as embeddings, target, predicted as list or json
