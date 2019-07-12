from examples.Utils import *
import multiprocessing
import random


def add_negative_samples(samples):
    all_title_set = set()
    for sample in samples:
        all_title_set.add(sample[0])
    result = []
    for sample in samples:
        result.append(sample)
        neg_samples = random.sample(all_title_set, 2)
        for neg_sample in neg_samples:
            result.append([neg_sample, sample[1], 0])
    return result


if __name__ == '__main__':
    all_samples = []
    with open('../samples/QLP/UrlMSAdtitle1TrainingData_1000.tsv', 'r', encoding='utf8') as f:
        for line in f:
            title, lp = line.split('\t')
            title = title.strip()
            lp = lp.strip()
            all_samples.append([title, lp, 1])

    random.shuffle(all_samples)

    train_ratio = 0.8
    train_data = all_samples[:100000]
    print(len(train_data))

    tasks = chunk(train_data, 20)

    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(add_negative_samples, tasks)
    train_data = []
    for sub_result in sub_results:
        train_data += sub_result
    random.shuffle(train_data)
    with open('../samples/QLP/data/train.tsv', 'w', encoding='utf8') as f:
        for sample in train_data:
            f.write(sample[0] + '\t' + sample[1] + '\t' + str(sample[2]) + '\n')

    test_data = all_samples[-100000:-80000]

    tasks = chunk(test_data, 20)

    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(add_negative_samples, tasks)
    test_data = []
    for sub_result in sub_results:
        test_data += sub_result
    random.shuffle(test_data)
    with open('../samples/QLP/data/test.tsv', 'w', encoding='utf8') as f:
        for sample in test_data:
            f.write(sample[0] + '\t' + sample[1] + '\t' + str(sample[2]) + '\n')

    dev_data = all_samples[-10000:]

    tasks = chunk(dev_data, 20)

    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(add_negative_samples, tasks)
    dev_data = []
    for sub_result in sub_results:
        dev_data += sub_result
    random.shuffle(dev_data)
    with open('../samples/QLP/data/dev.tsv', 'w', encoding='utf8') as f:
        for sample in dev_data:
            f.write(sample[0] + '\t' + sample[1] + '\t' + str(sample[2]) + '\n')
