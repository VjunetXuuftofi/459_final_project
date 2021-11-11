import argparse
import requests


def call_classifier(text, labels):
    response = requests.post("http://127.0.0.1:5000/classify", json={"text": text, "labels": labels})
    return response.json()


def process_text(filepath, labels):
    print("Loading file...")
    with open(filepath, "r") as f:
        text = "/n".join(f.readlines())
    labels = labels.split(",")
    print("Classifying text...")
    results = call_classifier(text, labels)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", default="test.txt")
    parser.add_argument("--labels", default="anger,excitement,happiness,racism")
    args = parser.parse_args()
    process_text(args.file_dir, args.labels)