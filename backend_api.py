from flask import Flask, jsonify, request
from sentence_transformers import CrossEncoder
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer

# initialize our Flask application
app = Flask(__name__)
USE_CROSS_ENCODER = True
USE_FEW_SHOT_GPT = False

encoder_options = ['microsoft/deberta-v2-xxlarge-mnli', 'facebook/bart-large-mnli', 'microsoft/deberta-v2-xlarge-mnli',
                   'roberta-large-mnli', "valhalla/distilbart-mnli-12-9"]

# Use entailment models to make the prediction. --------------------------------------------------------------------

if USE_CROSS_ENCODER:
    model = CrossEncoder('roberta-large-mnli')  # Use smaller model if you are using too much RAM. Roberta is not too large.


def cross_encoder_entailment(info):
    scores = model.predict(
        [[info["text"], label] for label in info["labels"]])
    return softmax(scores, axis=1)[:, 1]


# Another approach: few-shot learning with a GPT model. Try GPT-J, which has better performance. -----------------------
if USE_FEW_SHOT_GPT:
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


def get_score_of_sequence(input_ids, sequence_ids):
    working_input = input_ids
    gen_tokens = model.generate(input_ids, max_length=working_input.shape[1] + sequence_ids.shape[1],
                                output_scores=True, return_dict_in_generate=True, top_k=1)
    if tokenizer.decode(sequence_ids[0]) in tokenizer.decode(gen_tokens.sequences[0][working_input.shape[1]:]):
        return 1
    return 0


def few_shot_entailment(info):
    predictions = {}
    for label in info["labels"]:
        prompt = ""
        for example, truth in info["few_shot"][label]:
            prompt += example + " -> " + (label if truth else "not " + label) + "\n"
        prompt += info["text"] + " -> "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        positive_class = tokenizer(label, return_tensors="pt").input_ids

        positive_score = get_score_of_sequence(input_ids, positive_class)

        predictions[label] = positive_score
    return predictions

# TODO: Test other classification methods!


@app.route("/classify", methods=["POST"])
def classify():
    if request.method == 'POST':
        req = request.get_json()
        if USE_CROSS_ENCODER:
            results = cross_encoder_entailment(req)
        if USE_FEW_SHOT_GPT:
            results = few_shot_entailment(req)

        return jsonify(results)


if __name__ == '__main__':
    app.run()