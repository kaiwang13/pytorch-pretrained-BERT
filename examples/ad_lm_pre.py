from functools import reduce

from pytorch_pretrained_bert import OpenAIGPTTokenizer

from examples.Utils import *
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from pandas import Series


def filtration(samples):
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    result = []

    for sample in samples:
        total = 0
        zeros = 0
        title1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample[0]))
        title2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample[1]))
        description = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample[2]))
        for id in title1 + title2 + description:
            if id == 0:
                zeros += 1
            total += 1
        if 1.0 * zeros / total < 0.1 and len(title1 + title2 + description) <= 60:
            result.append(sample)
    return result


def generate_negative_samples(eval):
    all_title2 = set()
    all_description = set()
    for sample in eval:
        all_title2.add(sample[1])
        all_description.add(sample[2])
    all_title2 = list(all_title2)
    all_description = list(all_description)

    result = []
    for sample in eval:
        result.append(sample)
        result.append([sample[0], random.choice(all_title2), sample[2]])
        result.append([sample[0], sample[1], random.choice(all_description)])
        result.append([sample[0], random.choice(all_title2), random.choice(all_description)])
    return result


if __name__ == '__main__':
    samples = []
    with open('../samples/QLP/UrlMSWholeAdtitleTrainingData_1000.tsv', 'r', encoding='utf8') as input_file:
        for line in input_file:
            line = line.strip('\n')
            title1, title2, description, _ = line.split('\t')
            samples.append([title1, title2, description])
    tasks = chunk(samples, 20)
    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(filtration, tasks)
    samples = reduce(lambda x, y: x + y, sub_results)
    # #
    # # zero_ratio = []
    # # length = []
    # # for results in sub_results:
    # #     zero_ratio += results[0]
    # #     length += results[1]
    # #
    # # pkl_dump(zero_ratio, '../samples/LM/zero_ratio.pkl')
    # # pkl_dump(length, '../samples/LM/length.pkl')
    #
    # length = pkl_load('../samples/LM/length.pkl')
    # # zero_ratio = pkl_load('../samples/LM/zero_ratio.pkl')
    # values, base = np.histogram(length, bins=50)
    # # evaluate the cumulative
    # cumulative = np.cumsum(values)
    # plt.plot(base[:-1], cumulative, c='blue')
    # # length = pkl_load('../samples/LM/length.pkl')
    #
    # # sns.distplot(zero_ratio, hist=True)
    # # sns.distplot(length, hist=True)
    #
    # plt.show()
    random.shuffle(samples)
    train = samples[:-1000]
    eval = generate_negative_samples(samples[-1000:])
    json_dump(samples, '../samples/LM/Ads/data/filtration/all_ads.json')
    json_dump(train, '../samples/LM/Ads/data/filtration/train.json')
    json_dump(eval, '../samples/LM/Ads/data/filtration/eval.json')
