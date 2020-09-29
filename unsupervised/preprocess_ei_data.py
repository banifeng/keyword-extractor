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
        cur = {}
        while i < n and datas[i] != '\n':
            s = datas[i]
            if s.startswith('Title:'):
                cur['title'] = s[6:]
            if s.startswith('Abstract:'):
                cur['abstract'] = s[9:]
            i += 1
        i += 1
        if len(cur) == 2:
            res.append(cur)
    print(res)