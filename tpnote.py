#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:04:24 2022

@author: chebblidisdier


Partie 1 : Préparation des données
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import log2

df = pd.read_csv("synthetic.csv")
"""
nb0 = df['Class'].value_counts()
print(nb0)
"""
"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['Attr_K'], df['Attr_L'])
plt.xlabel('Attr_K')
plt.ylabel('Attr_L')
"""
print(df)

test = df.groupby(['Class','Attr_A'])
test.size()

"""
maxc = df['Attr_N'].loc[df['Attr_N'].idxmax()]
minc = df['Attr_N'].loc[df['Attr_N'].idxmin()]
print(minc)
print(maxc)

A : min 3.84 max 17.63
B : min 3.73 max 16.26
C : min 409.40 max 1725.27
D : min 65.04 max 131.32
E : min 26.74 max 176.65
F : min 192.89 max 1666.64
G : min 6.70 max 12.68
H : min 2.04 max 16.56
I : min 1.29 max 16.77
J : min 21.56 max 160.82
K : min 666.06 max 1369.16
L : min 6.34 max 13.34
M : min 654.66 max 1361.13
N : min 34.54 max 155.13

Recuperer une valeur associee a une etiquette donnee
occ_setosa = series . get ( ’ Iris - setosa ’)
"""

"""
2. Arbre de décision
"""

def entropie_df(df):
    res = 0
    nbl = df.shape[0]
    maxdf = df['Class'].loc[df['Class'].idxmax()]
    for ind in range(maxdf+1):
        val = df[df.Class == ind].shape[0]/nbl
        res -= val * log2(val)
    return res    

def info_gain(df, i):
    sump = 0
    ent = entropie_df(df)
    nbl = shape[0]
    p = df[df.Class == ind].shape[0]/nbl
    maxdf = df['Class'].loc[df['Class'].idxmax()]
    for ind in range (maxdf+1):
        part = df[df.Class == ind]
        sump += (p/nbl)*entropie_df(part)
        print(sump)
        print(entropie_df(part))
    gain = ent - sump
    print(gain)

ent = entropie_df(df)
info_gain()















