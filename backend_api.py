from flask import Flask, jsonify, request
from sentence_transformers import CrossEncoder
from scipy.special import softmax

# initialize our Flask application
app = Flask(__name__)

encoder_options = ['microsoft/deberta-v2-xxlarge-mnli', 'facebook/bart-large-mnli', 'microsoft/deberta-v2-xlarge-mnli',
                   'roberta-large-mnli', "valhalla/distilbart-mnli-12-9"]

model = CrossEncoder(encoder_options[0]) # Use smaller model if you are using too much RAM. Roberta is not too large.


def cross_encoder_entailment(text, labels):
    model = CrossEncoder('microsoft/deberta-v2-xxlarge-mnli')
    scores = model.predict(
        [[text, label] for label in labels])
    return softmax(scores, axis=1)[:, 1]

# TODO: Test other classification methods!


@app.route("/classify", methods=["POST"])
def classify():
    if request.method == 'POST':
        req = request.get_json()
        results = cross_encoder_entailment(req["text"], req["labels"])
        output = {}
        for label, result in zip(req["labels"], results):
            output[label] = float(result)
        return jsonify(output)


if __name__ == '__main__':
    app.run()