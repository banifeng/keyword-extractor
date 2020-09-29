# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com
@time: 2020-03-16 16:39
"""
from operator import itemgetter
from collections import defaultdict
import os
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
class KeywordExtractor(object):

    STOP_WORDS = set((
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
        "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
        "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
    ))

    def set_stop_words(self, stop_words_path):
        # abs_path = _get_abs_path(stop_words_path)
        abs_path = stop_words_path
        if not os.path.isfile(abs_path):
            raise Exception("jieba: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)

    def extract_tags(self, *args, **kwargs):
        raise NotImplementedError

class UndirectWeightedGraph:
    d = 0.85
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)
        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        sorted_keys = sorted(self.graph.keys())
        for x in range(10):  # 10 iters
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s
        return ws

class TextRank(KeywordExtractor):

    def __init__(self):
        self.stop_words = self.STOP_WORDS.copy()
        self.span = 10

    def pairfilter(self, wp):
        # 单词不是停用词
        # 词性在词性表中
        # 单词长度大于1
        return (wp[1] in self.pos_filt and len(wp[0].strip()) > 1
                and wp[0].lower() not in self.stop_words)

    def textrank(self, sentence, topK=14, withWeight=False, allowPOS=('JJ', 'JJR', 'JJS', 'NNS', 'NN', 'NNP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'IN'), withFlag=False):
        sentence = sentence.replace('- ', '').lower()
        print(sentence)
        self.pos_filt = frozenset(allowPOS)
        g = UndirectWeightedGraph()
        cm = defaultdict(int)
        words = tuple(nltk.pos_tag(nltk.word_tokenize(sentence)))
        for i, wp in enumerate(words):
            if self.pairfilter(wp):
                for j in range(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pairfilter(words[j]):
                        continue
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp[0], words[j][0])] += 1

        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)
        out = set()
        max_i = len(words)
        i = 0
        tags = tags[:topK]
        while i<max_i:
            if words[i][0] in tags:
                cur = [words[i][0]]
                while i+1<max_i and words[i+1][0] in tags:
                    i += 1
                    cur.append(words[i][0])
                if len(cur) > 1 and 'N' in words[i][1]:
                    out.add(' '.join(cur))
            i += 1
        if topK:
            return out, tags[:topK]
        else:
            return out, tags

    extract_tags = textrank

text = """
Commonsense reasoning is a long-standing challenge for deep learning. For exam- ple, it is difficult to use neural networks to tackle the Winograd Schema dataset [1]. In this paper, we present a simple method for commonsense reasoning with neural networks, using unsupervised learning. Key to our method is the use of language models, trained on a massive amount of unlabled data, to score multiple choice ques- tions posed by commonsense reasoning tests. On both Pronoun Disambiguation and Winograd Schema challenges, our models outperform previous state-of-the-art methods by a large margin, without using expensive annotated knowledge bases or hand-engineered features. We train an array of large RNN language models that operate at word or character level on LM-1-Billion, CommonCrawl, SQuAD, Gutenberg Books, and a customized corpus for this task and show that diversity of training data plays an important role in test performance. Further analysis also shows that our system successfully discovers important features of the context that decide the correct answer, indicating a good grasp of commonsense knowledge.
"""
text_rank = TextRank()
import time
s = time.time()
# for i in range(1000):
res = text_rank.extract_tags(text)
print(res)
print(time.time()-s)