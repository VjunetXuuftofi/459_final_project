import argparse
import requests
import json


def call_classifier(info):
    response = requests.post("http://127.0.0.1:5000/classify", json=info)
    return response.json()


def process_text(filepath, labels):
    print("Loading file...")
    with open(filepath) as f:
        info = json.load(f)
    print("Classifying text...")
    results = call_classifier(info)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", default="input.json")
    parser.add_argument("--labels", default="anger,excitement,happiness,racism")
    args = parser.parse_args()
    process_text(args.file_dir, args.labels)