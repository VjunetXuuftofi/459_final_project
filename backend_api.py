from flask import Flask, jsonify, request
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np

# initialize our Flask application
app = Flask(__name__)
USE_MNLI = False
USE_FEW_SHOT_GPT = False
USE_EMBEDDING = True
EXPERIMENTING = False

# Use NLI models to make the prediction. --------------------------------------------------------------------

def mnli_entailment(paragraphs, info):
    scores = {}
    for label in info["labels"]:
        predictions = []
        for paragraph in paragraphs:
            predictions.append(model.predict([paragraph, label]))
        predictions = np.array(predictions)
        scores[label] = [float(i) for i in softmax(predictions[:,[0,2]], axis=1)[:,1]]
    return scores

# Use GPT models to make the prediction. --------------------------------------------------------------------

def get_score_of_sequence(input_ids, sequence_ids):
    working_input = input_ids
    gen_tokens = model.generate(input_ids, max_length=working_input.shape[1] + sequence_ids.shape[1],
                                output_scores=True, return_dict_in_generate=True, top_k=1)
    if tokenizer.decode(sequence_ids[0]) in tokenizer.decode(gen_tokens.sequences[0][working_input.shape[1]:]):
        return 1
    return 0


def gpt_completion(paragraphs, info):
    scores = {}
    for label in info["labels"]:
        scores[label] = []
        for paragraph in paragraphs:
            prompt = ""
            for example, truth in info["few_shot"][label]:
                prompt += example + " -> " + (label if truth else "not " + label) + "\n"
            prompt += info["text"] + " -> "
            input_ids = tokenizer(paragraph, return_tensors="pt").input_ids
            positive_class = tokenizer(label, return_tensors="pt").input_ids

            positive_score = get_score_of_sequence(input_ids, positive_class)

            scores[label].append(positive_score)
    return scores

# Use embedding models to make the comparison.  --------------------------------------------------------------------

def embedding_similarity(paragraphs, info):
    encoding = embedder.encode(paragraphs)
    scores = {}
    for label in info["labels"]:
        scores[label] = [float(i) for i in util.dot_score(embedder.encode(label), encoding)[0]]
    return scores


def split_paragraphs(info):
    start_index = 0
    paragraphs = []
    indices = []
    for i, c in enumerate(info["text"]):
        if c == "\n":
            if i != start_index:
                paragraphs.append(info["text"][start_index:i])
                indices.append((start_index, i))
                start_index = i + 1
            else:
                start_index += 1
    if start_index != len(info["text"]) - 1:
        paragraphs.append(info["text"][start_index:])
        indices.append((start_index, len(info["text"])))
    return paragraphs, indices


@app.route("/classify", methods=["POST"])
def classify():
    if request.method == 'POST':
        req = request.get_json()

        paragraphs, indices = split_paragraphs(req)

        if EXPERIMENTING:
            if random.randint(0, 1):
                print("MNLI model")
                scores = mnli_entailment(paragraphs, req)
            else:
                print("Embedding Model")
                scores = embedding_similarity(paragraphs, req)
        else:
            if USE_MNLI:
                scores = mnli_entailment(paragraphs, req)
            if USE_FEW_SHOT_GPT:
                scores = gpt_completion(paragraphs, req)
            else:
                scores = embedding_similarity(paragraphs, req)

        results = {}
        for label in req["labels"]:
            results[label] = sorted(zip(paragraphs, scores[label], indices), key=lambda x: x[1], reverse=True)

        return jsonify(results)


if __name__ == '__main__':
    if EXPERIMENTING:
        USE_MNLI = True
        USE_EMBEDDING = True

    if USE_FEW_SHOT_GPT:
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    if USE_EMBEDDING:
        embedder = SentenceTransformer("sentence-transformers/msmarco-bert-base-dot-v5")

    if USE_MNLI:
        model = CrossEncoder('facebook/bart-large-mnli')



    app.run()