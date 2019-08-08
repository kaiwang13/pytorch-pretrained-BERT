import pathlib


if __name__ == '__main__':
    pathlib.Path('../samples/QLP/data/title/full/forAether/').mkdir(parents=True, exist_ok=True)
    samples = []
    i = 0
    with open('../samples/QLP/data/title/full/train.tsv', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            query, doc, label = line.split('\t')
            samples.append([label, ' ', str(i), query, doc])
            i += 1
    with open('../samples/QLP/data/title/full/forAether/train.tsv', 'w', encoding='utf8') as f:
        for sample in samples:
            f.write('\t'.join(sample) + '\n')

    samples = []
    i = 0
    with open('../samples/QLP/data/title/full/dev.tsv', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            query, doc, label = line.split('\t')
            samples.append([label, ' ', str(i), query, doc])
            i += 1
    with open('../samples/QLP/data/title/full/forAether/dev.tsv', 'w', encoding='utf8') as f:
        for sample in samples:
            f.write('\t'.join(sample) + '\n')