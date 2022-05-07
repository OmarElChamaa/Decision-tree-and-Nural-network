#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:04:24 2022

@author: chebblidisdier et el chamaa

Partie 1 : Préparation des données
"""
from tkinter.messagebox import RETRY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import log2
from sklearn.model_selection import train_test_split
from scipy import stats

"""
Max et min des attributs des données

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

"""

#Fonction permettant de trouver l'index d'une ligne via la valeur d'un attribut
#Elle nous sert notamment pour passer de la valeur de quantile() à l'index pour partitionner
def row_to_index(df, quart, a):
    for i in range (len(df)):
        if (df.iloc[i].at[a] == quart):
            return i
    return -1


#Calcul de l'entropie 
def entropie_df(df) : 
    nb_lignes = df.shape[0]
    series = df['Class'].value_counts()
    res = 0 
    for i in series :
        p = i/nb_lignes
        res += p*log2(p)
    return -res 


#pour calculer le gain d'un attribut 
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

### Construction de l'arbre 
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
        print(node.node_result(spacing))
        #print(spacing + node.node_result(spacing))
        return
    print ('{}[Attribute : {} Split value :{}]'.format(spacing,node.attribute,node.split))
    print (spacing + ' > True ')
    print_tree(node.lbranch,spacing + '-')
    print (spacing + ' > False ')
    print_tree(node.rbranch,spacing + '-')
    return

#Fonction permettant d'obtenir la prédiction d'une donnée selon un arbre de décision
#node : arbre de décision
#instance : donnée à inférer
def inference(node, instance):
    if node.leaf:
        #On retourne la classe avec la prédiction la plus haute du noeud
        return node.pred.axes[0].array[0]
    else :
        value = instance[node.attribute]
        if value < node.split :
            return inference(node.lbranch, instance)
        else :
            return inference(node.rbranch, instance)
