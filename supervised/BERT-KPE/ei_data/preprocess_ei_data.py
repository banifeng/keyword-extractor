# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com
@time: 2020-09-23 09:42
"""
with open('ei_01.txt', 'r', encoding='utf-8') as f:
    datas = f.readlines()
    res = []
    n = len(datas)
    i = 0
    while i < n:
        if datas[i] == '\n':
            i += 1
            continue
        cur = {'keyword': ''}
        while i < n and datas[i] != '\n':
            s = datas[i]
            if s.startswith('Title:'):
                cur['title'] = s[6:].rstrip()
            if s.startswith('Abstract:'):
                cur['abstract'] = s[9:].rstrip()
            i += 1
        i += 1
        if len(cur) == 3:
            res.append(cur)
    print(res[-1])
import json
with open('ei.json', 'w', encoding='utf-8') as f:
    for r in res:
        f.write(json.dumps(r, ensure_ascii=False))
        f.write('\n')