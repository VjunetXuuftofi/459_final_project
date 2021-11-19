import argparse
import requests
import json
import nltk.data
import nltk
import numpy as np
from sklearn import metrics

from pathlib import Path
import matplotlib.pyplot as plt
import math
import os

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def call_classifier(info):
    response = requests.post("http://127.0.0.1:5000/classify", json=info)
    return response.json()


def process_text(filepath, labels, model):
    print("Loading file...")
    with open("test/"+filepath, encoding='utf-8') as f:
        info = json.load(f)

    result_file = "{}-{}".format(model, Path(filepath).stem)
    output = open("results/{}/".format(model) + result_file + "-result.txt", "w+", encoding='utf-8')
    probs = get_probs(output, info)

    target = []
    for arr in info["gt"]:
        target += arr
    fpr, tpr, thresholds = calc_rates(probs, target)

    index = get_best_threshold_index(fpr, tpr)

    plot_roc_curve(fpr, tpr, model, result_file, index, thresholds[index])


def test_all(model):
    probs = []
    target = []

    for filename in os.listdir("test/amazon-reviews"):
        with open("test/amazon-reviews/" + filename, encoding='utf-8') as f:
            info = json.load(f)

        for arr in info["gt"]:
            target += arr

        probs += get_probs(None, info)
    
    fpr, tpr, thresholds = calc_rates(probs, target)

    index = get_best_threshold_index(fpr, tpr)
    plot_roc_curve(fpr, tpr, model, "Amazon", index, thresholds[index])


def get_best_threshold_index(fpr, tpr):
    min_dist = math.inf
    for i in range(len(tpr)):
        dist = np.linalg.norm(np.array((0, 1)) - np.array((fpr[i], tpr[i])))
        if dist < min_dist:
            index = i
            min_dist = dist
    return index


def get_probs(output, info):
    print("Classifying text...")
    probs = []
    # for line in tokenizer.tokenize(info["text"]):
    #     print(line)
    #     info["text"] = line
    results = call_classifier(info)
    print(results)
    if output:
        output.write(info["text"] + "\n")
        output.write(str(results) + "\n")
    probs += list(results.values())
    return probs


def calc_rates(probs, target):
    thresholds = [x/100.0 for x in range(0,101,2)]
    fpr = []
    tpr = []
    
    target = np.array(target)
    for t in thresholds:
        prediction = np.zeros(len(probs))
        prediction[np.array(probs) > t] = 1
        prediction = prediction.astype(int)

        current_tpr = np.sum(prediction & target) / np.sum(target)
        current_fpr = np.sum(prediction & (1 - target)) / np.sum(1 - target)

        tpr.append(current_tpr)
        fpr.append(current_fpr)
    return fpr, tpr, thresholds


def plot_roc_curve(fpr, tpr, model, result_file, index, best_threshold):
    plt.plot(fpr, tpr)
    plt.scatter(fpr[index], tpr[index], s=20, c='r')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    auc = metrics.auc(fpr, tpr)
    plt.title('ROC {} t={} AUC={}'.format(result_file, best_threshold, auc))
    plt.savefig("results/{}/roc_plots/{}.png".format(model, result_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", default="input.json")
    parser.add_argument("--labels", default="anger,excitement,happiness,racism")
    parser.add_argument("--model", default="deberta-v2-xxlarge")
    parser.add_argument("--test_all", default="False")
    args = parser.parse_args()
    if args.test_all == "True":
        test_all(args.model)
    else:
        process_text(args.file_dir, args.labels, args.model)