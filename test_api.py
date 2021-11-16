import argparse
import requests
import json
import nltk.data
import nltk
from pathlib import Path

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def call_classifier(info):
    response = requests.post("http://127.0.0.1:5000/classify", json=info)
    return response.json()


def process_text(filepath, labels, model):
    print("Loading file...")
    with open("test/"+filepath, encoding='utf-8') as f:
        info = json.load(f)
    output = open("results/{}/{}-{}-result.txt".format(model, model, Path(filepath).stem), "w+", encoding='utf-8')

    print("Classifying text...")
    for line in tokenizer.tokenize(info["text"]):
        print(line)
        info["text"] = line
        results = call_classifier(info)
        print(results)
        output.write(line + "\n")
        output.write(str(results) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", default="input.json")
    parser.add_argument("--labels", default="anger,excitement,happiness,racism")
    parser.add_argument("--model", default="deberta-v2-xxlarge")
    args = parser.parse_args()
    process_text(args.file_dir, args.labels, args.model)