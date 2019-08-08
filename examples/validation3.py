from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
import pathlib
from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from examples.run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features, \
    compute_metrics

from examples.BertDataSet import BertDataSet

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)

from examples.Utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def padding(list, max_len):
    if len(list) < max_len:
        list = list + [0 for _ in range(max_len - len(list))]
    return list


def preprocess(query_tokens, doc_tokens, tokenizer, max_len):
    if len(query_tokens) + len(doc_tokens) + 3 > max_len:
        tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + doc_tokens[:max_len - len(query_tokens) - 3] + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 for _ in range(len(tokens))]
        token_type_ids = [0 for _ in range(len(query_tokens) + 2)] + [1 for _ in
                                                                      range(max_len - len(query_tokens) - 2)]
    else:
        tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 for _ in range(len(tokens))]
        token_type_ids = [0 for _ in range(len(query_tokens) + 2)] + \
                         [1 for _ in range(len(tokens) - len(query_tokens) - 2)]

    return padding(input_ids, max_len), padding(input_mask, max_len), padding(token_type_ids, max_len)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    sep = ' [SEP] '
    samples = []
    i = 0
    input = {}
    with open('../samples/QLP/OutputDataOfGoodDebug.tsv', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            label, title1, title2, description, doc = line.split('\t')
            url = doc.split()[0]
            samples.append([i, title1 + sep + title2 + sep + description, url, doc])
            input[i] = [int(label), -1, title1 + sep + title2 + sep + description, url, doc]
            i += 1
    # json_dump(samples, '../samples/QLP/ValidationData_after.json')
    # samples = json_load('../samples/QLP/ValidationData_after.json')
    epoch_output_dir = '/home/work/waka/projects/pytorch-pretrained-BERT/samples/QLP/result/base/fullAd/Random/2parts/epoch2'
    model = BertForSequenceClassification.from_pretrained(epoch_output_dir, num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained(epoch_output_dir, do_lower_case=True)

    def pre_in_batch_eval(raw):
        index = []
        title = []
        url = []
        doc = []
        input_ids = []
        input_mask = []
        segment_ids = []
        for sample in raw:
            index.append(sample[0])
            title.append(sample[1])
            url.append(sample[2])
            doc.append(sample[3])
            one_input_ids, one_input_mask, one_segment_ids = preprocess(tokenizer.tokenize(sample[1]),
                                                                        tokenizer.tokenize(sample[3]),
                                                                        tokenizer, 128)
            input_ids.append(one_input_ids)
            input_mask.append(one_input_mask)
            segment_ids.append(one_segment_ids)

        return index, \
               title, \
               url, \
               doc, \
               torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(input_mask, dtype=torch.long), \
               torch.tensor(segment_ids, dtype=torch.long)


    test_loader = DataLoader(
        BertDataSet(samples),
        batch_size=500,
        shuffle=False,
        collate_fn=pre_in_batch_eval,
        num_workers=5
    )
    model.eval()
    result = {}
    with tqdm(total=len(test_loader)) as t:
        for batch_idx, (index, title, url, doc, input_ids, input_mask, segment_ids) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            predicted_label = torch.argmax(logits, dim=-1).cpu().numpy()
            for i in range(len(index)):
                result[index[i]] = predicted_label[i].item()
            t.update()
    total = 0
    correct = 0
    for k in input.keys():
        input[k][1] = result[k]
        total += 1
        if input[k][0] == result[k]:
            correct += 1
    print(100.0 * correct / total)
    json_dump(input, '../samples/QLP/self_bert_fullAd_2parts_epoch2_label_data.json')
