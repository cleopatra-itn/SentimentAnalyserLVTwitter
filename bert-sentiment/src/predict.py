import datetime
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from absl import app, flags, logging
from loguru import logger
from scipy import stats
from sklearn import metrics, model_selection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter

import config
import dataset
import engine
from model import BERTBaseUncased
from utils import categorical_accuracy, label_decoder, label_encoder

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
flags.DEFINE_string('test_file', None, "")
flags.DEFINE_string('model_path', None, "")

FLAGS = flags.FLAGS


def main(_):
    test_file = config.DATASET_LOCATION + "eval.prep.test.csv"
    model_path = config.MODEL_PATH
    if FLAGS.test_file:
        test_file = FLAGS.test_file
    if FLAGS.model_path:
        model_path = FLAGS.model_path
    df_test = pd.read_csv(test_file).fillna("none")

    # Commenting as there are no labels
    if FLAGS.features:
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

    device = config.device
    
    model = BERTBaseUncased()
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device(device)))
    model.to(device)

    outputs, extracted_features = engine.predict_fn(
        test_data_loader, model, device, extract_features=FLAGS.features)
    df_test["predicted"] = outputs
    # save file
    df_test.to_csv(model_path.split(
        "/")[-2]+'.csv', header=None, index=False)

    if FLAGS.features:
        pca = PCA(n_components=50, random_state=7)
        X1 = pca.fit_transform(extracted_features)
        tsne = TSNE(n_components=2, perplexity=10, random_state=6,
                    learning_rate=1000, n_iter=1500)
        X1 = tsne.fit_transform(X1)
        # if row == 0: print("Shape after t-SNE: ", X1.shape)

        X = pd.DataFrame(np.concatenate([X1], axis=1),
                         columns=["x1", "y1"])
        X = X.astype({"x1": float, "y1": float})

        # Plot for layer -1
        plt.figure(figsize=(20, 15))
        p1 = sns.scatterplot(x=X["x1"], y=X["y1"], palette="coolwarm")
        # p1.set_title("development-"+str(row+1)+", layer -1")
        x_texts = []
        for output, value in zip(outputs, df_test.label.values):
            if output == value:
                x_texts.append("@"+label_decoder(output)
                               [0] + label_decoder(output))
            else:
                x_texts.append(label_decoder(value) +
                               "-" + label_decoder(output))

        X["texts"] = x_texts
        # X["texts"] = ["@G" + label_decoder(output) if output == value else "@R-" + label_decoder(value) + "-" + label_decoder(output)
        #               for output, value in zip(outputs, df_test.label.values)]

        # df_test.label.astype(str)
        #([str(output)+"-" + str(value)] for output, value in zip(outputs, df_test.label.values))
        # Label each datapoint with the word it corresponds to
        for line in X.index:
            text = X.loc[line, "texts"]+"-"+str(line)
            if "@U" in text:
                p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                        size='medium', color='blue', weight='semibold')
            elif "@P" in text:
                p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                        size='medium', color='green', weight='semibold')
            elif "@N" in text:
                p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                        size='medium', color='red', weight='semibold')
            else:
                p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text, horizontalalignment='left',
                        size='medium', color='black', weight='semibold')
        plt.show()
        plt.savefig(model_path.split(
            "/")[-2]+'-figure.svg', format="svg")
        # loocv = model_selection.LeaveOneOut()
        # model = KNeighborsClassifier(n_neighbors=8)
        # results = model_selection.cross_val_score(model, X, Y, cv=loocv)
        # for i, j in outputs, extracted_features:
        #     utils.write_embeddings_to_file(extracted_features, outputs)


if __name__ == "__main__":
    app.run(main)
