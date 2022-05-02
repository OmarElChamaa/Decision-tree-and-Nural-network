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

# test = df.groupby(['Class','Attr_A'])
# test.size()

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

#split donnees 

traindf = df.quantile(0.75)
print(traindf)
# testdf =    
# print(testdf)
#Calcul de l'entropie :|
def entropie_df(df) : 
    nb_lignes = df.shape[0]
    series = df['Class'].value_counts()
    res = 0 
    for i in series :
        p = i/nb_lignes
        res -= p*log2(p)
    return res 


#pour calculer le gain d'un attribut :)
def info_gain(df, a):
    sump = 0
    ent = entropie_df(df)
    split_value = 0 
    partitions = [None,None]
    sorted_data = df.sort_values(by = a)
    classe = sorted_data["Class"].iloc[0]
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


# pour calculer le meilleur attribut :D
def super_gain(df,attributes) :
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

print('\n\n\033[1mGain : \033[0m',info_gain(df,'Attr_A')[0])
print('\n\n\033[1mMeilleur Gain : \033[0m \n\n',super_gain(df,df.columns.tolist()[0:14]))

#d'apres ce calcul, le meilleur attribut est f avec un gaine de 0.007733553206961563 et max split est  de 488.5951205915278 !!

### Construction de l'arbre :(

class Node():
    def __init__(self,split = None , attribut = None, feuille = False , filsGauche = None , filsDroit = None , pred = None) : 
        self.split = split 
        self.attribut = attribut
        self.feuille = feuille
        self.filsGauche = filsGauche
        self.filsDroit = filsDroit 
        self.pred = pred 



# def ze_tree(df,depth,target,attributes) :
#     attributes , gain , split , partitions = super_gain(df,attributes) 
#     pred = df[target].value_counts()
    


# ent = entropie_df(df)
# info_gain()















