from examples.Utils import *
import multiprocessing
import random
import pathlib


def add_negative_samples(samples):
    all_title1_set = set()
    all_title2_set = set()
    all_description_set = set()
    all_lp_set = set()

    for sample in samples:
        all_title1_set.add(sample[0])
        all_title2_set.add(sample[1])
        all_description_set.add(sample[2])
        all_lp_set.add(sample[3])

    all_title1_set = list(all_title1_set)
    all_title2_set = list(all_title2_set)
    all_description_set = list(all_description_set)
    all_lp_set = list(all_lp_set)

    result = []
    sep = ' '
    for sample in samples:
        result.append([sample[0] + sep + sample[1] + sep + sample[2], sample[3], '1'])
        neg_sample = [random.choice(all_title1_set),
                      random.choice(all_title2_set),
                      random.choice(all_description_set),
                      random.choice(all_lp_set)]

        result.append([neg_sample[0] + sep + sample[1] + sep + sample[2], sample[3], '0'])
        result.append([sample[0] + sep + neg_sample[1] + sep + sample[2], sample[3], '0'])
        result.append([sample[0] + sep + sample[1] + sep + neg_sample[2], sample[3], '0'])
        result.append([sample[0] + sep + sample[1] + sep + sample[2], neg_sample[3], '0'])
    return result


if __name__ == '__main__':
    pathlib.Path('../samples/QLP/data/fullAd/Random/2parts_noSEP/').mkdir(parents=True, exist_ok=True)
    all_samples = []
    with open('../samples/QLP/UrlMSWholeAdtitleTrainingData_1000.tsv', 'r', encoding='utf8') as f:
        for line in f:
            title1, title2, description, lp = line.split('\t')
            title1 = title1.strip()
            title2 = title2.strip()
            description = description.strip()
            lp = lp.strip()
            all_samples.append([title1, title2, description, lp, '1'])

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
    with open('../samples/QLP/data/fullAd/Random/2parts_noSEP/train.tsv', 'w', encoding='utf8') as f:
        for sample in train_data:
            f.write('\t'.join(sample) + '\n')

    test_data = all_samples[-100000:-80000]

    tasks = chunk(test_data, 20)

    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(add_negative_samples, tasks)
    test_data = []
    for sub_result in sub_results:
        test_data += sub_result
    random.shuffle(test_data)
    with open('../samples/QLP/data/fullAd/Random/2parts_noSEP/test.tsv', 'w', encoding='utf8') as f:
        for sample in test_data:
            f.write('\t'.join(sample) + '\n')

    dev_data = all_samples[-10000:]

    tasks = chunk(dev_data, 20)

    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(add_negative_samples, tasks)
    dev_data = []
    for sub_result in sub_results:
        dev_data += sub_result
    random.shuffle(dev_data)
    with open('../samples/QLP/data/fullAd/Random/2parts_noSEP/dev.tsv', 'w', encoding='utf8') as f:
        for sample in dev_data:
            f.write('\t'.join(sample) + '\n')
