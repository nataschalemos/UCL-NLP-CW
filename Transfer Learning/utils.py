import os
import random
import logging

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds, source):
    print('Printing Lengths:')
    print(len(preds))
    print(len(labels))
    print('Printing preds')
    print(preds)
    print('Printing labels')
    print(labels)
    assert len(preds) == len(labels)
    results = dict()

    results["accuracy" + source] = accuracy_score(labels, preds)

    results["macro_precision_" + source], results["macro_recall_" + source], results[
        "macro_f1_" + source], _ = precision_recall_fscore_support(
        labels, preds, average="macro")

    results["micro_precision_" + source], results["micro_recall_" + source], results[
        "micro_f1" + source], _ = precision_recall_fscore_support(
        labels, preds, average="micro")

    results["weighted_precision_" + source], results["weighted_recall_" + source], results[
        "weighted_f1_" + source], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")

    return results
