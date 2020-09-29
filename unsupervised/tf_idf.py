# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com
@time: 2020-03-09 15:48
"""
# step1:分词，并用Counter记录词频
# step2:将Counter写入文件
# step3:针对每个句子进行测试
import json
import jieba
from collections import Counter
import os
import math

class KeyWordsAbstract(object):

    def __init__(self, ei_file):
        self.ei_file = ei_file
        self.df_file = 'tf_counter.json'
        if os.path.exists(self.df_file):
            print('df_file exists, loading the df_file ...')
            self.df = json.load(open(self.df_file))
        else:
            self.df = Counter()
            self._read_ei_file()
        self.jieba_tokenizer = jieba.Tokenizer()
        self.jieba_tokenizer.tmp_dir = '.'

    def _read_ei_file(self):
        if os.path.exists(self.df_file):
            print('warning: df_file exists, jump!')
            return
        abstract_texts = []
        print('loading the ei_file...')
        with open(self.ei_file, 'r', encoding='utf-8') as fin:
            line = fin.readline()
            while line:
                if line.startswith('Abstract'):
                    abstract_texts.append(line.lower().split(':', maxsplit=1)[1])
                line = fin.readline()
        print('constructing document frequency...')

        for text in abstract_texts:
            self.df.update(set(self.jieba_tokenizer.cut(text)))
        self.df['document_num@value'] = len(abstract_texts)
        with open(self.df_file, 'w', encoding='utf-8') as fout:
            json.dump(obj=self.df, fp=fout, ensure_ascii=False, indent=4)

    def chooseTopKWords(self, text, k=3):
        tokens = [w for w in self.jieba_tokenizer.cut(text.lower()) if w]
        counter = Counter(tokens)
        res = []
        n = len(tokens)
        for key, value in counter.items():
            tf = value/n
            idf = math.log(self.df['document_num@value']//(self.df.get(key, 0)+1)+1, 10)
            score = tf * idf
            res.append([key, score])
        res.sort(key=lambda x:x[1], reverse=True)
        return res[:k]

text = "Abstract:The stability and synchronization analysis of two chaotic Rulkov maps coupled by bidirectional and symmetric chemical synapses are taken into account. As a function of intrinsic control parameters &alpha;, &sigma;, &eta;, reversal potential v, synaptic parameters &theta;, k, and external chemical coupling strength g<inf>c</inf>, conditions for stability of a fixed point for this system are derived. Some typical domains are chosen for numerical simulations which include time evolution of transmembrane voltages and phase portraits, and both of them are presented for theoretical analysis. Based on the master stability functions approach and calculation of the maximum Lyapunov exponents of synchronization errors, synchronized regions of the coupled neurons and a strip-shaped chaotic structure in parameter-space are obtained. Specially, given some values of control parameter &alpha;, we propose interval ranges of coupling strength g<inf>c</inf>in which the two chaotic Rulkov map-based neurons can be synchronized completely. It is shown that there exist different transition mechanisms of the neuronal spiking and bursting synchronization. The synchronized regions will become smaller and smaller as control parameter &alpha; or synaptic parameter &theta; increases. Nevertheless, the coupled neurons can at first transit from desynchrony to in-phase synchronization, and then to complete synchronization as chemical coupling strength g<inf>c</inf>increases. Compared with control parameter &alpha; and synaptic parameter &theta;, chemical coupling strength g<inf>c</inf>plays an opposite role in the process of synchronization transition. These findings could be useful for further understanding the role of two chaotic Rulkov maps coupled by bidirectional and symmetric chemical synapses in the field of cooperative behaviors of coupled neurons. &copy; 2015 Elsevier B.V."

text = """
Commonsense reasoning is a long-standing challenge for deep learning. For exam- ple, it is difficult to use neural networks to tackle the Winograd Schema dataset [1]. In this paper, we present a simple method for commonsense reasoning with neural networks, using unsupervised learning. Key to our method is the use of language models, trained on a massive amount of unlabled data, to score multiple choice ques- tions posed by commonsense reasoning tests. On both Pronoun Disambiguation and Winograd Schema challenges, our models outperform previous state-of-the-art methods by a large margin, without using expensive annotated knowledge bases or hand-engineered features. We train an array of large RNN language models that operate at word or character level on LM-1-Billion, CommonCrawl, SQuAD, Gutenberg Books, and a customized corpus for this task and show that diversity of training data plays an important role in test performance. Further analysis also shows that our system successfully discovers important features of the context that decide the correct answer, indicating a good grasp of commonsense knowledge.
"""
abstractor = KeyWordsAbstract(ei_file='ei_01.txt')

res = abstractor.chooseTopKWords(text, k=5)
print('\n\n关键词抽取结果如下：')
for key, score in res:
    print(key, ":", '%.5f'%score)