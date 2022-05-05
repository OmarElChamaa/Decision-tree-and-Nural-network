#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:04:24 2022

@author: chebblidisdier


Partie 1 : Préparation des données
"""
from tkinter.messagebox import RETRY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import log2
from sklearn.model_selection import train_test_split

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
print("Nos attributs" , df.columns.tolist()[:-1])

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


#df.quantile(0.25) et on compare avec 0.75 restant  
#df.quantile(0.50) et on compare avec 0.5 restant 
#df.quantile(0.75) et on compare avec 0.25 restant
#au lieu de faire partitions 
#entropie premnier quatile et compare au prochain 


def row_to_index(df, quart, a):
    for i in range (len(df)):
        if (df.iloc[i].at[a] == quart):
            return i
    return -1

#pour calculer le gain d'un attribut :)
#
def info_gain_quart(df, a):
    sump = 0
    ent = entropie_df(df)
    max_gain = 0 
    max_split = 0
    partitions = [None,None]
    max_partitions = [None, None]
    sorted_data = df.sort_values(by = a) 
    for i in range(3): 
        quartile = 0.25 + i*0.25
        quartile_val = sorted_data[a].quantile(quartile,interpolation='nearest')
        quartile_ind = row_to_index(sorted_data, quartile_val, a)
        partitions[0] = sorted_data.iloc[:quartile_ind]
        partitions[1] = sorted_data.iloc[quartile_ind:]
        sump = 0 
        for x in partitions:
            sump += len(x)/len(df) * entropie_df (x)
        gain = ent - sump 
        #print("Attribut ", a, ": gain = ",gain, " , split = ", quartile_val)
        if (gain>max_gain) : 
            max_gain = gain 
            max_partitions = partitions
            max_split = quartile_val
    
    return max_gain , max_split , max_partitions
    


# pour calculer le meilleur attribut :D
def super_attribute(df,attributes) :
    max_gain = -1 
    partitions = []
    max_split = 0 
    attribute = ''
    for i in range (len(attributes)) :
        gain,split,tmpPartitions = info_gain_quart(df,attributes[i])
        if(gain>max_gain):
            max_gain = gain 
            partitions = tmpPartitions
            max_split = split 
            attribute = attributes[i]
    return attribute,max_gain,max_split,partitions

# print('\n\n\033[1mGain : \033[0m',info_gain_quart(df,'Attr_A')[:2])
# print('\n\n\033[1mMeilleur Gain : \033[0m \n\n',super_attribute(df,df.columns.tolist()[:-1]))

#d'apres ce calcul, le meilleur attribut est b avec un gaine de x et max split est  de y !!

### Construction de l'arbre :(

class Node():
    def __init__(self,split = None , attribute = None, lbranch = None , rbranch = None , pred = None, leaf = False) : 
        self.split = split 
        self.attribute = attribute
        self.leaf = leaf
        self.lbranch = lbranch
        self.rbranch = rbranch 
        self.pred = pred 
    def node_result(self, spacing=''):
        s = ''
        for v in range(len(self.pred.values)):
            s += ' Class ' + str(self.pred.index[v]) + ' Count: ' + str(self.pred.values[v]) + '\n' + spacing
        return s




def ze_tree(df, cur_depth, target, attributes, max_depth) :
    attribute, gain , split , partitions = super_attribute(df,attributes) 
    pred = df[target].value_counts()
    if(cur_depth > max_depth or len(attributes) == 0 or gain == 0) :
        return Node(pred = pred , leaf = True)
    attributes.remove(attribute)
    lbranch = ze_tree(partitions[0], cur_depth+1, target, attributes, max_depth)  
    rbranch = ze_tree(partitions[1], cur_depth+1, target, attributes, max_depth)
    return Node(split, attribute, lbranch, rbranch, pred)
    


def print_tree(node,spacing = ' ') :
    if node is None :
        return
    if node.leaf :
        node.node_result(spacing)
        return
    print ('{}[Attribute : {} Split value :{}]'.format(spacing,node.attribute,node.split))
    print (spacing + ' > True ')
    print_tree(node.lbranch,spacing + '-')
    print (spacing + ' > False ')
    print_tree(node.rbranch,spacing + '-')
    return



def node_result(self, spacing=''):
    s = ''
    for v in range(len(self.pred.values)):
        s += ' Class ' + str(self.pred.index[v]) + ' Count: ' + str(self.pred.values[v]) + '\n' + spacing
    return s

    
# self.prediction: obtenu avec l'instruction data[cible].value_counts()

node = ze_tree(df,0,'Class',df.columns.tolist()[:-1],4)
#print_tree(node)

#Permet de donner l'évaluation du node feuille selon les prédictions
def eval_node(node, df) :
    predmat = {
              #0  1  2  3
        '0' : [0, 0, 0, 0],
        '1' : [0, 0, 0, 0],
        '2' : [0, 0, 0, 0],
        '3' : [0, 0, 0, 0]
    } 
    confusionMatrix = pd.DataFrame(predmat)

    nbok = 0
    for i in range(100):
        sample = df.sample()
        res , cpred=  inference(node, sample)
        label = sample.iat[0,-1]
        confusionMatrix.iat[label,cpred] += 1
    print(confusionMatrix)
    return 



def inference(node, instance):
    if node.leaf:
        return node.pred , node.pred.axes[0].array[0]
    else :
        value = instance[node.attribute].tolist()[0]
        if value < node.split :
            return inference(node.lbranch, instance)
        else :
            return inference(node.rbranch, instance)

print("---------- inférence wouhou ------------")

sample = df.sample()
eval_node(node,df)











