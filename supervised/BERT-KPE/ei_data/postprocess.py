# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com
@time: 2020-09-23 20:00
"""

data = '../data/prepro_dataset/kp20k/kp20k.eval.json'
label = '../results/test_bert2span_kp20k_roberta_09.23_19.30/predictions/bert2span.kp20k_eval.roberta.epoch_5.checkpoint'
import json
with open(data, 'r', encoding='utf-8') as f:
    with open(label, 'r', encoding='utf-8') as fout:
        datas = f.readlines()
        labels = fout.readlines()
        for d, l in zip(datas, labels):
            text = json.loads(d)['doc_words']
            text = ' '.join(text)
            print(text)
            key = json.loads(l)['KeyPhrases']
            for k in key:
                print(' '.join(k))
            print()