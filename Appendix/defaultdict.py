# -*- coding:utf-8 -*-
from collections import defaultdict

a = defaultdict(int)
a ['6'] = 56789
for key ,vlaue in a.items():
    print("key ",key,"  vlaue ",vlaue)