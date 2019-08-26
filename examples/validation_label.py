from examples.Utils import *
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import average_precision_score


if __name__ == '__main__':
    results = json_load('/home/work/waka/projects/pytorch-pretrained-BERT/samples/LM/Ads/models/epoch1/eval_result_label.json')
    labels = np.array([abs(pair[3] - 1) for pair in results])
    scores = np.array([pair[3] for pair in results])
    average_precision = average_precision_score(labels, scores)
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR of relevant model on relevant label unbalance')
    plt.show()
