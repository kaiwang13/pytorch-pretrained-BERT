from functools import reduce

from pytorch_pretrained_bert.tokenization import BertTokenizer
from examples.Utils import *
import multiprocessing
import random


def segment(samples):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return [tokenizer.tokenize(sample) for sample in samples]


def pre(path, filename):
    samples = []

    with open(path + filename + '.tsv', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            samples.append(line)
    samples = random.sample(samples, min(len(samples), 1000000))

    tasks = chunk(samples, 20)

    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(segment, tasks)
    result = []
    for sub_result in sub_results:
        result += sub_result
    json_dump(result, path + filename + '_segmented.json')
    length_sum = reduce(lambda x, y: x + y, map(lambda sample: len(sample), result), 0)
    print(filename + ' done. Total lines: ' + str(len(result)) + '. Average length: ' + str(1.0 * length_sum / len(result)))


if __name__ == '__main__':
    pre('../../samples/QLP/data/Stat/', 'AllUrl')
    pre('../../samples/QLP/data/Stat/', 'AllFullAdtitles')
    pre('../../samples/QLP/data/Stat/', 'AllMS10')
    pre('../../samples/QLP/data/Stat/', 'AllMS17')
    pre('../../samples/QLP/data/Stat/', 'AllMS22')
