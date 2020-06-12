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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['interactive'] == True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()
logger.add("experiment.log")

flags.DEFINE_boolean('features', True, "")
FLAGS = flags.FLAGS


def main(_):

    test_file = config.DATASET_LOCATION + "eval.prep.test.csv"
    df_test = pd.read_csv(test_file).fillna("none").head(51)
    # Commenting as there are no labels
    df_test.label = df_test.label.apply(label_encoder)

    logger.info(f"Bert Model: {config.BERT_PATH}")
    logger.info(
        f"Current date and time :{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
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
    model.load_state_dict(torch.load(
        config.MODEL_PATH, map_location=torch.device(device)))
    model.to(device)

    outputs, extracted_features = engine.predict_fn(
        test_data_loader, model, device, extract_features=FLAGS.features)
    if FLAGS.features:
        pca = PCA(n_components=50, random_state=7)
        X1 = pca.fit_transform(extracted_features)
        tsne = TSNE(n_components=2, perplexity=10, random_state=6,
                    learning_rate=1000, n_iter=1500)
        X1 = tsne.fit_transform(X1)
        # if row == 0: print("Shape after t-SNE: ", X1.shape)

        # # Recording the position of the tokens, to be used in the plot
        # position = np.array(list(span))
        # position = position.reshape(-1,1)

        X = pd.DataFrame(np.concatenate([X1], axis=1),
                         columns=["x1", "y1"])
        X = X.astype({"x1": float, "y1": float})

        # Plot for layer -1
        plt.figure(figsize=(20, 15))
        p1 = sns.scatterplot(x=X["x1"], y=X["y1"], palette="coolwarm")
        # p1.set_title("development-"+str(row+1)+", layer -1")
        X["texts"] = df_test.label.astype(str)
        #([str(output)+"-" + str(value)] for output, value in zip(outputs, df_test.label.values))
        # Label each datapoint with the word it corresponds to
        for line in X.index:
            text = X.loc[line, "texts"]
            if "@P" in text:
                p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                        size='medium', color='blue', weight='semibold')
            elif "@G" in text:
                p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                        size='medium', color='green', weight='semibold')
            elif "@R" in text:
                p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                        size='medium', color='red', weight='semibold')
            else:
                p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text, horizontalalignment='left',
                        size='medium', color='black', weight='semibold')
        plt.show()
        plt.savefig('figure.png') 

        # for i, j in outputs, extracted_features:
        #     utils.write_embeddings_to_file(extracted_features, outputs)


if __name__ == "__main__":
    app.run(main)
