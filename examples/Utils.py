import json
import pickle
import math


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def json_dump(obj, path):
    json.dump(obj, open(path, 'w', encoding='utf8'), indent=4, ensure_ascii=False, sort_keys=True, cls=SetEncoder)


def json_load(path):
    return json.load(open(path, 'r', encoding='utf8'))


def pkl_dump(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def pkl_load(path):
    return pickle.load(open(path, 'rb'))


def chunk(list, n):
    result = []
    for i in range(n):
        result.append(list[math.floor(i / n * len(list)):math.floor((i + 1) / n * len(list))])
    return result
