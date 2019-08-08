from examples.Utils import *
import multiprocessing
import random


def add_negative_samples(samples):
    all_title_set = set()
    for sample in samples:
        all_title_set.add(sample[0])
    all_title_set = list(all_title_set)
    result = []
    for sample in samples:
        result.append(sample)
        neg_samples = random.sample(all_title_set, 2)
        for neg_sample in neg_samples:
            result.append([neg_sample, sample[1], 0])
    return result


if __name__ == '__main__':
    all_samples = []
    with open('../samples/QLP/OutputDataOfRelevant.tsv', 'r', encoding='utf8') as f:
        for line in f:
            label, _, _, title, lp = line.split('\t')
            title = title.strip()
            lp = lp.strip()
            all_samples.append([title, lp, label])

    random.shuffle(all_samples)
    training = all_samples[:-1000]
    positive = 0
    negative = 0
    for sample in training:
        if sample[2] == '1':
            positive += 1
        else:
            negative += 1
    print(positive)
    print(negative)

    evaluation = all_samples[-1000:]

    with open('../samples/QLP/data/fullAd/Relevant/train.tsv', 'w', encoding='utf8') as f:
        for sample in training:
            f.write('\t'.join(sample) + '\n')

    with open('../samples/QLP/data/fullAd/Relevant/dev.tsv', 'w', encoding='utf8') as f:
        for sample in evaluation:
            f.write('\t'.join(sample) + '\n')