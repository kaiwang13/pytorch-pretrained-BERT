from examples.Utils import *


if __name__ == '__main__':
    label_data = []
    grammarOK_map = {
        'good': 1,
        'fair': 1,
        'embrassing': 0,
        'bad': 0
    }
    humanlike_map = {
        'yes': 1,
        'no': 0
    }
    accurate_map = {
        'yes': 1,
        'no': 0
    }
    relevant_map = {
        'yes': 1,
        'no': 0
    }
    grammar_bad = []
    humanlike_bad = []

    with open('../samples/LM/Ads/data/label_full.tsv', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip('\t')
            _, _, title1, title2, _, description, _, _, _, grammarOK, humanlike, accurate, relevant, comment1, comment2 = line.split('\t')
            if grammarOK != 'notscorable':
                label_data.append([title1, title2, description,
                                   comment1 + ' ' + comment2,
                                   grammarOK_map[grammarOK],
                                   humanlike_map[humanlike],
                                   accurate_map[accurate],
                                   relevant_map[relevant]])
                if grammarOK_map[grammarOK] == 0:
                    grammar_bad.append([title1, title2, description, comment1.strip() + ' ' + comment2.strip()])
                if humanlike_map[humanlike] == 0:
                    humanlike_bad.append([title1, title2, description, comment1.strip() + ' ' + comment2.strip()])
    json_dump(label_data, '../samples/LM/Ads/data/label_full.json')
    json_dump(grammar_bad, '../samples/LM/Ads/data/grammar_bad.json')
    json_dump(humanlike_bad, '../samples/LM/Ads/data/humanlike_bad.json')
