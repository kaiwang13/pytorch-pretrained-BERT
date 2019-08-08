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
    with open('../samples/QLP/SampleData.tsv', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            index, title1, title2, description, _, _, url, doc = line.split('\t')
            samples.append([index, title1 + sep + title2 + sep + description, url, doc])
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
            scores = F.softmax(logits, dim=-1)[:, 1:].squeeze().cpu().numpy()
            for i in range(len(index)):
                if index[i] not in result:
                    result[index[i]] = []
                result[index[i]].append([title[i], url[i], doc[i], scores[i].item()])
            t.update()
    top1_result = {}
    for k, v in result.items():
        top1_title = ''
        top1_url = ''
        top1_score = -1.0
        for sample in v:
            if sample[3] > top1_score:
                top1_title = sample[0]
                top1_url = sample[1]
                top1_score = sample[3]
        top1_result[k] = {
            'Title': top1_title,
            'Url': top1_url
        }
    json_dump(top1_result, '../samples/QLP/self_bert_fullAd_2parts_epoch2_top1.json')
    json_dump(result, '../samples/QLP/self_bert_fullAd_2parts_epoch2.json')
