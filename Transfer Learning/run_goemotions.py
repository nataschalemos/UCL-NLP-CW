import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict

from transformers import (
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from model import BertForMultiLabelClassification
from utils import (
    init_logger,
    set_seed,
    compute_metrics
)
from data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)

logger = logging.getLogger(__name__)


def train(args,
          model,
          tokenizer,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total
    )

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
                "sources": batch[4]
            }
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_original = None
    preds_group = None
    out_label_ids_original = None
    out_label_ids_group = None
    #out_sources = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
                "sources": batch[4]
            }
            outputs = model(**inputs)
            tmp_eval_loss, original_predictions, group_predictions = outputs

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        # get original predictions and labels
        if original_predictions.numel() != 0:
            if preds_original is None:
                preds_original = 1 / (1 + np.exp(-original_predictions.detach().cpu().numpy()))  # Sigmoid
                sources_vec = inputs["sources"].detach().cpu().numpy()
                original_idx = sources_vec[sources_vec == 0]
                out_label_ids_all = inputs["labels"].detach().cpu().numpy()
                out_label_ids_original = out_label_ids_all[original_idx.astype(int),:28]
            else:
                preds_original = np.append(preds_original, 1 / (1 + np.exp(-original_predictions.detach().cpu().numpy())), axis=0)  # Sigmoid
                sources_vec = inputs["sources"].detach().cpu().numpy()
                original_idx = sources_vec[sources_vec == 0]
                out_label_ids_all = inputs["labels"].detach().cpu().numpy()
                out_label_ids_original = np.append(out_label_ids_original, out_label_ids_all[original_idx.astype(int),:28], axis=0)

        # get group predictions and labels
        if group_predictions.numel() != 0:
            if preds_group is None:
                preds_group = 1 / (1 + np.exp(-group_predictions.detach().cpu().numpy()))  # Sigmoid
                sources_vec = inputs["sources"].detach().cpu().numpy()
                group_idx = sources_vec[sources_vec == 1]
                out_label_ids_all = inputs["labels"].detach().cpu().numpy()
                out_label_ids_group = out_label_ids_all[group_idx.astype(int),28:]
            else:
                preds_group = np.append(preds_group, 1 / (1 + np.exp(-group_predictions.detach().cpu().numpy())), axis=0)  # Sigmoid
                sources_vec = inputs["sources"].detach().cpu().numpy()
                group_idx = sources_vec[sources_vec == 1]
                out_label_ids_all = inputs["labels"].detach().cpu().numpy()
                out_label_ids_group = np.append(out_label_ids_group, out_label_ids_all[group_idx.astype(int),28:], axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }

    # # separate individual task predictions based on the source of the datapoints
    # preds_corrected = []
    # for i in range(len(preds)):
    #     if out_sources[i] == 0:
    #         preds_source = preds[i]
    #         preds_source[28:] = 0
    #         preds_corrected.append(preds_source)
    #     elif out_sources[i] == 1:
    #         preds_source = preds[i]
    #         preds_source[:28] = 0
    #         preds_corrected.append(preds_source)
    # preds_corrected = np.asarray(preds_corrected)

    preds_original[preds_original > args.threshold] = 1
    preds_original[preds_original <= args.threshold] = 0
    result_original = compute_metrics(out_label_ids_original, preds_original, "original")
    results.update(result_original)

    final_preds_group = np.zeros(preds_group.shape)
    for i in range(preds_group.shape[0]):
        final_preds_group[i, np.argmax(preds_group[i, :])] = 1
    result_group = compute_metrics(out_label_ids_group, final_preds_group, "group")
    results.update(result_group)

    # preds_group[preds_group > args.threshold] = 1
    # preds_group[preds_group <= args.threshold] = 0
    # result_group = compute_metrics(out_label_ids_group, preds_group, "group")
    # results.update(result_group)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return results


def main():
    # Read from config file and make args
    config_filename = "mtl.json"
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

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels_original=len(label_list_original),
        num_labels_group =len(label_list_group),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_list_all)},
        label2id={label: i for i, label in enumerate(label_list_all)},
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    model = BertForMultiLabelClassification.from_pretrained(
        args.model_name_or_path,
        config=config
    )

    # GPU or CPU
    #args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Load dataset
    train_dataset_sentiment = load_and_cache_examples(args, tokenizer, mode="train", source=1) if args.train_file else None
    dev_dataset_sentiment = load_and_cache_examples(args, tokenizer, mode="dev", source=1) if args.dev_file else None
    test_dataset_sentiment = load_and_cache_examples(args, tokenizer, mode="test", source=1) if args.test_file else None

    if dev_dataset_sentiment is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset_sentiment, dev_dataset_sentiment, test_dataset_sentiment)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))
        
    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True)))
        
    model = BertForMultiLabelClassification.from_pretrained(checkpoints[-1])
    model.to(args.device)
    
    files = glob.glob('/'+args.ckpt_dir+'/*')
    for f in files:
        os.remove(f)
        
    # Load dataset
    train_dataset_emotion = load_and_cache_examples(args, tokenizer, mode="train", source=0) if args.train_file else None
    dev_dataset_emotion = load_and_cache_examples(args, tokenizer, mode="dev", source=0) if args.dev_file else None
    test_dataset_emotion = load_and_cache_examples(args, tokenizer, mode="test", source=0) if args.test_file else None

    if dev_dataset_emotion is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset_emotion, dev_dataset_emotion, test_dataset_emotion)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = BertForMultiLabelClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset_emotion, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':

    main()
