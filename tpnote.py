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

# def entropie_df(df):
#     res = 0
#     nbl = df.shape[0]
#     maxdf = df['Class'].loc[df['Class'].idxmax()]
#     for ind in range(maxdf+1):
#         val = df[df.Class == ind].shape[0]/nbl
#         res -= val * log2(val)
#     return res    


def entropie_df(df) : 
    nb_lignes = df.shape[0]
    series = df['Class'].value_counts()
    res = 0 
    for i in series :
        p = i/nb_lignes
        res -= p*log2(p)
    return res 

print( entropie_df(df))
data_sorted = df.sort_values(by ='Attr_A')
print(data_sorted)

def info_gain(df, a):
    sump = 0
    ent = entropie_df(df)
    split_value = 0 
    partitions = [None,None]
    sorted_data = df.sort_values(by = a)
    classe = sorted_data['Class'].iloc[0]
    for i in range (len(sorted_data)) : 
        if sorted_data["Class"].iloc[i] != classe : 
            split_value = sorted_data[a].iloc[i]
            partitions[0] = sorted_data.iloc[:i]
            partitions[1] = sorted_data.iloc[i:]
            sump = 0 
            for x in partitions :
                sump += len(x)/len(df) * entropie_df (x)
            gain = ent - sump 
            break
    return gain , split_value , partitions

def best_attribute(df,attributes) :
    max_gain = -1 
    partitions = []
    max_split = 0 
    attribute = ''
    for i in range (len(attributes)) :
        gain,split,tmpPartitions = info_gain(df,attributes[i])
        if(gain>max_gain):
            max_gain = gain 
            partitions = tmpPartitions
            max_split = split 
            attribute = attributes[i]
    return attribute,max_gain,max_split,partitions

print('\n\n\033 [1mGain : \033[0m',info_gain(df,'Attr_A')[0])
print('\n\n\033 [1mMeilleur Gain : \033[0m',best_attribute(df,df.columns.tolist()[0:4])[0])


    # nbl = shape[0]
    # p = df[df.Class == ind].shape[0]/nbl
    # maxdf = df['Class'].loc[df['Class'].idxmax()]
    # for ind in range (maxdf+1):
    #     part = df[df.Class == ind]
    #     sump += (p/nbl)*entropie_df(part)
    #     print(sump)
    #     print(entropie_df(part))
    # gain = ent - sump
    # print(gain)

# ent = entropie_df(df)
# info_gain()















