import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """ A single training/test example for simple sequence classification. """

    def __init__(self, guid, text_a, text_b, label, source):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.source = source

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label, source):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.source = source

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_length,
):
    processor = GoEmotionsProcessor(args)
    label_list_len0 = len(processor.get_labels(0))
    label_list_len1 = len(processor.get_labels(1))

    def convert_to_one_hot_label(label, source):
        one_hot_label = [0] * (label_list_len0 + label_list_len1)
        for l in label:
            if source == 1:
                l += 28
            one_hot_label[l] = 1
        return one_hot_label

    # def convert_to_one_hot_label(label, source):
    #     if source == 0:
    #         one_hot_label = [0] * label_list_len0
    #         for l in label:
    #             one_hot_label[l] = 1
    #         one_hot_label = one_hot_label + [0] * label_list_len1
    #     elif source == 1:
    #         one_hot_label = [0] * label_list_len1
    #         for l in label:
    #             one_hot_label[l] = 1
    #         one_hot_label = [0] * label_list_len0 + one_hot_label
    #     return one_hot_label

    # def convert_to_one_hot_label(label, source):
    #     if source == 0:
    #         label_list_len = label_list_len0
    #     elif source == 1:
    #         label_list_len = label_list_len1
    #     #label_list_len = globals()["label_list_len" + str(source)]
    #     one_hot_label = [0] * label_list_len
    #     for l in label:
    #         one_hot_label[l] = 1
    #     return one_hot_label

    labels = [convert_to_one_hot_label(example.label, example.source) for example in examples]
    sources = [example.source for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i], source=sources[i])
        features.append(feature)

    for i, example in enumerate(examples[:10]):
        logger.info("*** Example ***")
        logger.info("guid: {}".format(example.guid))
        logger.info("sentence: {}".format(example.text_a))
        logger.info("tokens: {}".format(" ".join([str(x) for x in tokenizer.tokenize(example.text_a)])))
        logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
        logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
        logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
        logger.info("label: {}".format(" ".join([str(x) for x in features[i].label])))
        logger.info("source: {}".format(" ".join([str(features[i].source)])))

    return features


class GoEmotionsProcessor(object):
    """Processor for the GoEmotions data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self, source):
        if source == 0:
            data_dir = self.args.original_data_dir
        elif source == 1:
            data_dir = self.args.group_data_dir
        labels = []
        with open(os.path.join(data_dir, self.args.label_file), "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.rstrip())
        return labels

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return f.readlines()

    def _create_examples(self, lines, set_type, source):
        """ Creates examples for the train, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = line.strip()
            items = line.split("\t")
            text_a = items[0]
            label = list(map(int, items[1].split(",")))
            if i % 5000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, source=source))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        # Load original dataset (source=0)
        logger.info("LOOKING AT {}".format(os.path.join(self.args.original_data_dir, file_to_read)))
        original_examples = self._create_examples(self._read_file(os.path.join(self.args.original_data_dir,
                                                           file_to_read)), mode, 0)

        # Load group dataset (source=1)
        logger.info("LOOKING AT {}".format(os.path.join(self.args.group_data_dir, file_to_read)))
        group_examples = self._create_examples(self._read_file(os.path.join(self.args.group_data_dir,
                                                           file_to_read)), mode, 1)

        return original_examples + group_examples


def load_and_cache_examples(args, tokenizer, mode):
    processor = GoEmotionsProcessor(args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.all_data_dir,
        "cached_{}_{}_{}_{}".format(
            str(args.task),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            mode
        )
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file")
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is available")
        features = convert_examples_to_features(
            args, examples, tokenizer, max_length=args.max_seq_len
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    all_sources = torch.tensor([f.source for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_sources)
    return dataset
