# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com
@time: 2020-03-30 16:20
"""
import os
def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def trans_one_item(text_file, ann_file):
    text = open(text_file, encoding='utf-8').readline().strip()
    anns = open(ann_file, encoding='utf-8').readlines()
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    # Split on whitespace so that different tokens may be attributed to their original position.
    for c in text:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    labels = ['O'] * len(doc_tokens)
    for ann in anns:
        if ann.startswith('T'):
            split_ann = ann.strip().split('\t')
            start, entity_len = int(split_ann[1].split()[-2].split(';')[0]), len(split_ann[-1].split())
            entity_index = char_to_word_offset[start]
            for i in range(entity_index, entity_index+entity_len):
                if i == entity_index:
                    if labels[i] == 'O':
                        labels[i] = 'B-KeyWords'
                else:
                    if i==len(labels):
                        continue
                    labels[i] = 'I-KeyWords'
    return doc_tokens, labels

def process_dir(dir, output_file):
    tokens, labels = [], []
    with open(output_file, 'w', encoding='utf-8') as fout:
        for file in os.listdir(dir):
            if file.endswith('.txt'):
                cur_tokens, cur_labels = trans_one_item(os.path.join(dir, file), os.path.join(dir, file.replace('.txt', '.ann')))
                for t, l in zip(cur_tokens, cur_labels):
                    fout.write(t+' '+l+'\n')
                fout.write('\n')

process_dir('./train2', 'train.txt')
process_dir('./dev', 'dev.txt')
# trans_one_item('./train2/S0010938X1530161X.txt', './train2/S0010938X1530161X.ann')

