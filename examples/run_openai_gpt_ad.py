# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
          --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import os
import multiprocessing
import random
import logging
from functools import reduce
import pathlib
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     OpenAIAdam, cached_path, WEIGHTS_NAME, CONFIG_NAME, OpenAIGPTLMHeadModel)
from examples.Utils import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == 0)


model_name = ''


def tokenize_and_encode(dataset):
    special_tokens = ['<BOA>', '<SEP>', '<EOA>']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
    for i in range(len(dataset)):
        dataset[i] = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][0])),
                      tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][1])),
                      tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][2]))]

    return dataset


def tokenize_and_encode_single_part(dataset):
    special_tokens = ['<BOA>', '<EOA>']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
    for i in range(len(dataset)):
        dataset[i] = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][0])),
                      tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][1])),
                      tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][2]))]

    return dataset


def pre_process_datasets_single_part(encoded_datasets, input_len, start_token, end_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
        input_lengths = np.zeros((n_batch,), dtype=np.int64)
        lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
        for i, text, in enumerate(dataset):
            ad = [start_token] + text + [end_token]
            if len(ad) <= input_len:
                input_ids[i, :len(ad)] = ad
                input_lengths[i] = len(ad)
                lm_labels[i, :len(ad)] = ad
        all_inputs = (input_ids, input_lengths, lm_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets


def pre_process_datasets(encoded_datasets, input_len, start_token, sep_token, end_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
        input_lengths = np.zeros((n_batch,), dtype=np.int64)
        lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
        for i, (title1, title2, description), in enumerate(dataset):
            ad = [start_token] + title1 + [sep_token] + title2 + [sep_token] + description + [end_token]
            if len(ad) <= input_len:
                input_ids[i, :len(ad)] = ad
                input_lengths[i] = len(ad)
                lm_labels[i, :len(ad)] = ad
        all_inputs = (input_ids, input_lengths, lm_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument('--ft_name', type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument("--single_part", action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    if args.single_part:
        special_tokens = ['<BOA>', '<EOA>']
    else:
        special_tokens = ['<BOA>', '<SEP>', '<EOA>']

    model_name = args.model_name
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = OpenAIGPTLMHeadModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))
    model.to(device)
    original_model = model
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Encoding dataset...")
    train_dataset = json_load(args.train_dataset)
    eval_dataset = json_load(args.eval_dataset)

    tasks = chunk(train_dataset, 20)
    with multiprocessing.Pool(processes=20) as pool:
        if args.single_part:
            sub_results = pool.map(tokenize_and_encode_single_part, tasks)
        else:
            sub_results = pool.map(tokenize_and_encode, tasks)
    train_dataset = reduce(lambda x, y: x + y, sub_results)

    tasks = chunk(eval_dataset, 20)
    with multiprocessing.Pool(processes=20) as pool:
        if args.single_part:
            sub_results = pool.map(tokenize_and_encode_single_part, tasks)
        else:
            sub_results = pool.map(tokenize_and_encode, tasks)
    eval_dataset = reduce(lambda x, y: x + y, sub_results)

    encoded_datasets = (train_dataset, eval_dataset)

    # Compute the max input length for the Transformer
    input_length = args.max_seq_length

    # Prepare inputs tensors and dataloaders
    if args.single_part:
        tensor_datasets = pre_process_datasets_single_part(encoded_datasets, input_length, *special_tokens_ids)
    else:
        tensor_datasets = pre_process_datasets(encoded_datasets, input_length, *special_tokens_ids)
    train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)



    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                               lr=args.learning_rate,
                               warmup=args.warmup_proportion,
                               max_grad_norm=args.max_grad_norm,
                               weight_decay=args.weight_decay,
                               t_total=num_train_optimization_steps)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_lengths, lm_labels = batch
                loss = model(input_ids=input_ids, lm_labels=lm_labels)
                if n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])

            # Save a trained model
            if args.do_train:
                epoch_root = os.path.join(args.output_dir, args.ft_name, 'epoch' + str(epoch))
                pathlib.Path(epoch_root).mkdir(parents=True, exist_ok=True)
                # Save a trained model, configuration and tokenizer
                model_to_save = original_model.module if hasattr(original_model, 'module') else original_model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`
                output_model_file = os.path.join(epoch_root, WEIGHTS_NAME)
                output_config_file = os.path.join(epoch_root, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(epoch_root)

                # # Load a trained model and vocabulary that you have fine-tuned
                # model = OpenAIGPTLMHeadModel.from_pretrained(args.output_dir)
                # tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
                # model.to(device)

            if args.do_eval:
                original_model.eval()
                eval_loss, eval_accuracy, eval_ppl = 0, 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_lengths, lm_labels = batch
                    with torch.no_grad():
                        ppls = original_model.forward_ppl(input_ids, input_lengths)
                    if not args.single_part:
                        ppls = ppls.reshape(-1, 4).numpy()
                        tmp_eval_accuracy = accuracy(ppls)
                        eval_accuracy += tmp_eval_accuracy
                    else:
                        eval_ppl += ppls.sum()

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_loss = 1.0 * eval_loss / nb_eval_steps
                eval_ppl = 1.0 * eval_ppl / nb_eval_examples
                eval_accuracy = 1.0 * eval_accuracy / nb_eval_examples
                train_loss = 1.0 * tr_loss / nb_tr_steps if args.do_train else None

                result = {
                    'train_loss': train_loss
                }
                if args.single_part:
                    result['eval_ppl'] = eval_ppl
                else:
                    result['eval_accuracy'] = eval_accuracy

                output_eval_file = os.path.join(epoch_root, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
