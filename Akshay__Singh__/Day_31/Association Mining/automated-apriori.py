import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


# Finds the items that are infrequent
def has_infreq_subset(c, Lk_1):
    for item in c:
        k_1_set = c - frozenset([item])
        if ((k_1_set in Lk_1) == False):
            return True
    return False

# Generation of level-1 frequent itemsets
def find_freq_1_itemsets(D, f):
    S = []
    for row in D:
        S.append(set(row))  # remove duplicate
    item_count = defaultdict(int)
    for row in S:
        for item in row:
            item_count[item] += 1
    length = len(D)
    total = 0
    for row in D:
        total += len(row)
    op_sup = float(total) / (length * f)
    L = []
    for key, value in item_count.items():
        I = []
        if float(value) / length >= op_sup:
            I.append(key)
            L.append(frozenset(I))
    return L

# Generation of level-2 frequent itemsets
def apriori_gen(Lk_1, k):
    Ck = []
    for i in Lk_1:
        for j in Lk_1:
            c = i.union(j)
            if (len(c) == k):
                if (has_infreq_subset(c, Lk_1) == False):
                    Ck.append(c)
    remove_duplicate = set(Ck)
    o = list(remove_duplicate)
    return o

# Automated Apriori by calculating the minimum supports and progressive supports
def apriori(database, f):
    D = []
    for row in database:
        D.append(row)
    S = []
    for row in D:
        S.append(frozenset(row))
    number_of_transaction = len(S)
    L = []
    L.append(frozenset())  # L[0]
    L.append(find_freq_1_itemsets(D, f))  # L[1]
    p = L[1]
    e = defaultdict(int)

    for i in p:
        for row in S:
            if i.issubset(row):
                e[i] += 1  # count of level 1 items

    s = defaultdict(float)

    for i in e:
        s[i] = float(e[i]) / number_of_transaction  # support of level 1 items
    s1 = defaultdict(float)
    for i, j in s.items():
        m = list(i)
        s1[m[0]] = j

    min_conf = 0.7
    msd = []
    msd1 = defaultdict(float)
    csd = []
    csd.append(frozenset())
    csd.append(frozenset())
    css = defaultdict(float)
    mss = defaultdict(float)

    for i, j in s.items():
        msd1[i] = j * max(min_conf, j)
    msd.append(frozenset(msd1))
    k = 2
    C = []
    C.append(frozenset())  # C[0]
    C.append(frozenset())  # C[1]

    while (len(L[k - 1]) != 0):
        C.append(apriori_gen(L[k - 1], k))  # C[k]
        c_count = defaultdict(int)
        for c in C[k]:
            for x in S:
                if c.issubset(x):
                    c_count[c] += 1
        I = []
        csd1 = defaultdict(float)
        ms1 = defaultdict(float)

        for key, value in c_count.items():
            lmm = list(key)
            if k == 2:
                cs = 1
                for i in lmm:
                    cs = cs * s1[i]
                sup = float(value) / number_of_transaction
                ms = min(cs, sup)
                csd1[key], css[key] = cs, cs
                s[key] = sup
                ms1[key], mss[key] = ms, ms
            if k >= 3:
                cf = list(csd[k - 1])
                zf = []
                for q in cf:
                    if q.issubset(key):
                        zf.append(css[q])
                cs = min(zf)
                csd1[key], css[key] = cs, cs
                sup = float(value) / number_of_transaction
                s[key] = sup
                ms = min(cs, sup)
                ms1[key], mss[key] = ms, ms
        csd.append(frozenset(csd1))
        msd.append(frozenset(ms1))

        for key, value in c_count.items():
            if k == 2:
                lm = list(key)
                lk = list(csd[k])
                len_lk = len(lk)
                for h in lk:
                    if csd1[h] >= min(msd1[frozenset(lm[0])], msd1[frozenset(lm[1])]):
                        I.append(key)
            if k >= 3:
                cf = list(csd[k - 1])
                zf = []
                for q in cf:
                    if q.issubset(key):
                        zf.append(mss[q])
                if csd1[key] >= min(zf):
                    I.append(key)
        L.append(frozenset(I))  # L[k]
        k += 1
    return L, s


# Subset generation
def find_subset(item, l):
    h = []
    for i in range(1, l + 1):
        h.append(list(itertools.combinations(item, i)))
    g = []
    for i in h:
        for j in i:
            g.append(j)
    return g


# Generation of association rules using support and confidences of frequent itemsets
def association_rules(min_conf, suppotr1):
    rules = list()
    for item, supp in support1.items():
        l = len(item)
        if l > 1:
            subsets = find_subset(item, l)
            for A in subsets:
                B = item.difference(A)
                if B:
                    A = frozenset(A)
                    AB = A | B
                    confidence1 = support1[AB] / support1[A]
                    if confidence1 >= min_conf:
                        rules.append((A, B, confidence1))

    return rules


# Input dataset
itemset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
f = len(itemset)
database = [['F', 'N', 'L', 'M', 'E', 'A', 'H', 'O'], ['E', 'M', 'N', 'F', 'J', 'B', 'K', 'H', 'A'],
            ['I', 'D', 'B', 'L'], ['F', 'C', 'G', 'M', 'E', 'B', 'D', 'N'], ['J', 'B', 'L', 'O', 'D', 'E', 'L', 'F'],
            ['J', 'I', 'F', 'D', 'A', 'C'], ['D', 'E', 'G', 'H', 'B', 'K'], ['K', 'L'], ['N', 'B'],
            ['O', 'N', 'F', 'C', 'D', 'G', 'L', 'A']]
L, support1 = apriori(database, f)
x = frozenset([])
for i in L:
    if i == x:
        L.remove(i)
min_conf = 0.7
rule = association_rules(min_conf, support1)
h1 = []
for i in rule:
    g = i[0].union(i[1])
    h1.append(g)
s1 = set(h1)
y = {}
for i in s1:
    if i in support1:
        y[i] = support1[i]
c = []
for i in rule:
    d = i[0].union(i[1])
    c.append((d, i[2]))
c1 = set(c)
ff = []
for e in c1:
    if e[0] in y:
        ff.append((e[0], y[e[0]], e[1]))
so = pd.DataFrame(ff)
so.columns = ["ITEMS", "SUPPORT", "CONFIDENCE"]
print
so
print
"\n"

x = []
y = []
for i in ff:
    x.append(i[1])
    y.append(i[2])
print
"x  :  ", x
print
"y  :  ", y
colors = (1, 0, 0)
area = np.pi * 50

# Data visualization of the result
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()