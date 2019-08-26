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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def accuracy(out):
    outputs = np.argmin(out, axis=1)
    return np.sum(outputs == 0)


model_name = 'openai-gpt'


def tokenize_and_encode(dataset):
    special_tokens = ['<BOA>', '<SEP>', '<EOA>']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
    for i in range(len(dataset)):
        dataset[i] = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][0])),
                      tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][1])),
                      tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[i][2]))]

    return dataset


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
            if len(ad) <= 64:
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
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--max_input_length', type=int, default=64)
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

    special_tokens = ['<BOA>', '<SEP>', '<EOA>']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)

    eval_dataset = json_load(args.eval_dataset)
    eval_dataset = [[x[0], x[1], x[2]] for x in eval_dataset]

    tasks = chunk(eval_dataset, 20)
    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(tokenize_and_encode, tasks)
    eval_dataset = reduce(lambda x, y: x + y, sub_results)
    encoded_datasets = [eval_dataset]

    # Compute the max input length for the Transformer
    input_length = args.max_input_length  # Max size of input for the pre-trained model

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, *special_tokens_ids)
    eval_tensor_dataset = tensor_datasets[0]

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    token_log_freq = json_load('../samples/LM/token_log_freq.json')

    for epoch in range(2):
        epoch_root = os.path.join(args.output_dir, 'epoch' + str(epoch))
        model = OpenAIGPTLMHeadModel.from_pretrained(epoch_root)
        model.load_token_log_freq(token_log_freq)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(epoch_root)
        model.to(device)
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        results = json_load(args.eval_dataset)
        index = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_lengths, lm_labels = batch
            with torch.no_grad():
                ppls = model.forward_normalized_ppl(input_ids, input_lengths).numpy()
            for ppl in ppls:
                results[index].append(ppl.item())
                index += 1
        json_dump(results, os.path.join(args.output_dir, 'epoch' + str(epoch), 'eval_result_label_normalized_ppl.json'))
