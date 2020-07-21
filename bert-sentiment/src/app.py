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
from werkzeug.serving import run_simple
from werkzeug.wsgi import DispatcherMiddleware

class PrefixMiddleware(object):

    def __init__(self, app, prefix=''):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):

        if environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
            environ['SCRIPT_NAME'] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ["This url does not belong to the app.".encode()]


T = tokenizer.TweetTokenizer(preserve_handles=True, preserve_hashes=True, preserve_case=False, preserve_url=False)

app = Flask(__name__,
	static_folder='app_resources/static',
	static_url_path='/sentimentanalyzer', 
	instance_relative_config=True,  
	template_folder='app_resources/templates/public')
# app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='/sentimentanalyzer')
# app.config['SERVER_NAME'] = "cleopatra.ijs.si:1095"
# app.config['APPLICATION_ROOT'] = "/sentimentanalyzer/"


def simple(env, resp):
    resp(b'200 OK', [(b'Content-Type', b'text/plain')])
    return [b'Hello WSGI World']

# app.wsgi_app = DispatcherMiddleware(simple, {'/sentimentanalyzer/': app.wsgi_app})


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


@app.route("/sentimentanalyzer/predict", methods=['POST'])
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

@app.route("/sentimentanalyzer/")
def index():
    print("sending file...",file=sys.stdout)
    return render_template("index.html")

@app.route("/sentimentanalyzer/demo")
def demo():
    return render_template("demo.html")

@app.route("/sentimentanalyzer/models")
def models():
    return render_template("models.html")


if __name__ == "__main__":
    MODEL= BERTBaseUncased()
    MODEL.load_state_dict(torch.load(
        config.MODEL_PATH, map_location=torch.device(DEVICE)))
    MODEL.eval()
    
    app.run("0.0.0.0",port=1095, debug=True)
# host="http://cleopatra.ijs.si/sentimentanalyzer"