import json
import logging
import os
import glob

import torch

from attrdict import AttrDict

from transformers import BertTokenizer

from model import BertForMultiLabelClassification
from utils import (
    init_logger,
    set_seed
)
from data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)

from run_goemotions import evaluate

def evaluate_model(config_filename, checkpoint):

    logger = logging.getLogger(__name__)

    # Read from config file and make args
    #config_filename = "mtl.json"
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    processor = GoEmotionsProcessor(args)
    label_list_original = processor.get_labels(0)
    label_list_group = processor.get_labels(1)
    label_list_all = label_list_original + label_list_group

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Load dataset
    test_dataset_emotion = load_and_cache_examples(args, tokenizer, mode="test", source=0) if args.test_file else None

    results = {}
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    global_step = checkpoint.split("-")[-1]
    model = BertForMultiLabelClassification.from_pretrained(checkpoint)
    model.to(args.device)
    result = evaluate(args, model, test_dataset_emotion, mode="test", global_step=global_step)
    result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    results.update(result)