from pytorch_pretrained_bert.tokenization import BertTokenizer
from examples.Utils import *


if __name__ == '__main__':
    result = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    with open('../samples/QLP/data/tmp/DomainDiff1.tsv', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip('\n')
            url, _, _ = line.split('\t')
            result.append([url, ' '.join(tokenizer.tokenize(url))])
    json_dump(result, '../samples/QLP/data/tmp/result.json')