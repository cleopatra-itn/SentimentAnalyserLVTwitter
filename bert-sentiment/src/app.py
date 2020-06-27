import flask
import torch
from flask import Flask, render_template, request
from utils import label_full_decoder
import sys
import config
import dataset
import engine
from model import BERTBaseUncased
from tokenizer import tokenizer
T = tokenizer.TweetTokenizer(preserve_handles=True, preserve_hashes=True, preserve_case=False, preserve_url=False)

app = Flask(__name__, static_url_path='', static_folder='app/static',
            template_folder='app/templates/public')

MODEL = None
DEVICE = config.device

def preprocess(text):
    tokens = T.tokenize(text)
    print(tokens,file=sys.stderr)
    ptokens =[]
    for index, token in enumerate(tokens):
        if "@" in token:
            if index>0:
                # check if previous token was mention 
                if "@" in tokens[index-1]:
                    pass
                else:
                    ptokens.append("mention_0")
            else:
                ptokens.append("mention_0")
        else:
            ptokens.append(token)
        
    print(ptokens, file=sys.stderr)
    return " ".join(ptokens)

def sentence_prediction(sentence):
    sentence = preprocess(sentence)
    model_path = config.MODEL_PATH

    test_dataset = dataset.BERTDataset(
        review=[sentence],
        target=[0]
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

    outputs, [] = engine.predict_fn(test_data_loader, model, device)
    print(outputs)
    return outputs[0]
    # try:
    #     passexcept Exception as exp:
    #     print(exp, file=sys.stderr)


@app.route("/predict", methods=['POST'])
def predict():
    print(request.form, file=sys.stderr)
    # print([(x) for x in request.get_json()],file=sys.stderr)
    # sentence = request.get_json().get("sentence","")
    sentence = request.form['sentence']
    if sentence:
        print(sentence, file=sys.stderr)
        prediction= sentence_prediction(sentence)
        response= {}
        response["response"]= {
            'sentence': sentence,
            'prediction': label_full_decoder(prediction),
        }
        return flask.jsonify(response)
    else:
        return flask.jsonify({"error": "empty text"})


@ app.route("/")
def index():
    return render_template("index.html")


@ app.route("/demo")
def demo():
    return render_template("demo.html")


@ app.route("/models")
def models():
    return render_template("models.html")


if __name__ == "__main__":
    MODEL= BERTBaseUncased()
    MODEL.load_state_dict(torch.load(
        config.MODEL_PATH, map_location=torch.device(DEVICE)))
    MODEL.eval()
    app.run(debug=True)
