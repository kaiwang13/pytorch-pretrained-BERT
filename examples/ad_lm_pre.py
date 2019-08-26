from functools import reduce

from pytorch_pretrained_bert import OpenAIGPTTokenizer

from examples.Utils import *
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from pandas import Series
import pathlib


def token_stat(samples):
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    result = {}
    total_tokens = 0
    total_BOA_EOA = 0
    total_SEP = 0

    for sample in samples:
        title1 = tokenizer.tokenize(sample[0])
        title2 = tokenizer.tokenize(sample[1])
        description = tokenizer.tokenize(sample[2])
        all_tokens = title1 + title2 + description
        for token in all_tokens:
            if token not in result:
                result[token] = 0
            result[token] += 1
        total_tokens += len(all_tokens) + 4
        total_BOA_EOA += 1
        total_SEP += 2

    return result, total_tokens, total_BOA_EOA, total_SEP


def stat(samples):
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    title1_length = []
    title2_length = []
    description_length = []

    for sample in samples:
        title1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample[0]))
        title2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample[1]))
        description = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample[2]))
        title1_length.append(len(title1))
        title2_length.append(len(title2))
        description_length.append(len(description))

    return title1_length, title2_length, description_length


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


def output(samples, name):
    random.shuffle(samples)
    train = samples[:-1000]
    eval = samples[-1000:]
    root_path = '../samples/LM/Ads/data/' + name + '/'
    pathlib.Path(root_path).mkdir(parents=True, exist_ok=True)
    json_dump(samples, root_path + 'all_ads.json')
    json_dump(train, root_path + 'train.json')
    json_dump(eval, root_path + 'eval.json')


if __name__ == '__main__':
    samples = []
    with open('../samples/QLP/UrlMSWholeAdtitleTrainingData_1000.tsv', 'r', encoding='utf8') as input_file:
        for line in input_file:
            line = line.strip('\n')
            title1, title2, description, _ = line.split('\t')
            samples.append([title1, title2, description])
    tasks = chunk(samples, 20)
    with multiprocessing.Pool(processes=20) as pool:
        sub_results = pool.map(token_stat, tasks)

    result = {}
    total_tokens = 0
    total_BOA_EOA = 0
    total_SEP = 0
    for sub_result in sub_results:
        for k, v in sub_result[0].items():
            if k not in result:
                result[k] = 0
            result[k] += v
        total_tokens += sub_result[1]
        total_BOA_EOA += sub_result[2]
        total_SEP += sub_result[3]
    result['<BOA>'] = total_BOA_EOA
    result['<EOA>'] = total_BOA_EOA
    result['<SEP>'] = total_SEP
    print(total_tokens)
    json_dump(result, '../samples/LM/token_count.json')
    special_tokens = ['<BOA>', '<SEP>', '<EOA>']
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', special_tokens=special_tokens)

    log_freq = {}
    for k in result:
        log_freq[tokenizer.convert_tokens_to_ids(k)] = math.log(result[k]) - math.log(total_tokens)
        result[k] = 1.0 * result[k] / total_tokens
    json_dump(log_freq, '../samples/LM/token_log_freq.json')
    json_dump(total_tokens, '../samples/LM/total_tokens.json')
    json_dump(result, '../samples/LM/token_freq.json')

    #
    # pkl_dump(title1_length, '../samples/LM/title1_length.pkl')
    # pkl_dump(title2_length, '../samples/LM/title2_length.pkl')
    # pkl_dump(description_length, '../samples/LM/description_length.pkl')

    # title2_length = pkl_load('../samples/LM/title2_length.pkl')
    # # zero_ratio = pkl_load('../samples/LM/zero_ratio.pkl')
    # values, base = np.histogram(title2_length, bins=50)
    # # evaluate the cumulative
    # cumulative = np.cumsum(values)
    # plt.plot(base[:-1], cumulative, c='blue')
    # # length = pkl_load('../samples/LM/length.pkl')
    #
    # # sns.distplot(zero_ratio, hist=True)
    # # sns.distplot(length, hist=True)
    #
    # plt.show()

    # samples = []
    # with open('../samples/QLP/UrlMSWholeAdtitleTrainingData_1000.tsv', 'r', encoding='utf8') as input_file:
    #     for line in input_file:
    #         line = line.strip('\n')
    #         title1, title2, description, _ = line.split('\t')
    #         samples.append([title1, title2, description])
    # tasks = chunk(samples, 20)
    # with multiprocessing.Pool(processes=20) as pool:
    #     sub_results = pool.map(filtration, tasks)
    # samples = reduce(lambda x, y: x + y, sub_results)
    # # #
    # # # zero_ratio = []
    # # # length = []
    # # # for results in sub_results:
    # # #     zero_ratio += results[0]
    # # #     length += results[1]
    # # #
    # # # pkl_dump(zero_ratio, '../samples/LM/zero_ratio.pkl')
    # # # pkl_dump(length, '../samples/LM/length.pkl')
    # #
    # # length = pkl_load('../samples/LM/length.pkl')
    # # # zero_ratio = pkl_load('../samples/LM/zero_ratio.pkl')
    # # values, base = np.histogram(length, bins=50)
    # # # evaluate the cumulative
    # # cumulative = np.cumsum(values)
    # # plt.plot(base[:-1], cumulative, c='blue')
    # # # length = pkl_load('../samples/LM/length.pkl')
    # #
    # # # sns.distplot(zero_ratio, hist=True)
    # # # sns.distplot(length, hist=True)
    # #
    # # plt.show()
    # all_title1 = list(set(map(lambda x: x[0], samples)))
    # all_title2 = list(set(map(lambda x: x[1], samples)))
    # all_description = list(set(map(lambda x: x[2], samples)))
    #
    # output(all_title1, 'title1')
    # output(all_title2, 'title2')
    # output(all_description, 'description')
    #
    # # random.shuffle(samples)
    # # train = samples[:-1000]
    # # eval = generate_negative_samples(samples[-1000:])
    # # json_dump(samples, '../samples/LM/Ads/data/filtration/all_ads.json')
    # # json_dump(train, '../samples/LM/Ads/data/filtration/train.json')
    # # json_dump(eval, '../samples/LM/Ads/data/filtration/eval.json')
