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

"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['Attr_K'], df['Attr_L'])
plt.xlabel('Attr_K')
plt.ylabel('Attr_L')
"""
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


#Fonction permettant d'évaluer le modèle node sur un partie de df
#On crée la matrice de confusion du modèle
def eval_node(node, df, samplesize) :
    
    predmat = {
              #0  1  2  3
        '0' : [0, 0, 0, 0],
        '1' : [0, 0, 0, 0],
        '2' : [0, 0, 0, 0],
        '3' : [0, 0, 0, 0]
    } 
    confusionMatrix = pd.DataFrame(predmat)
    nbl = [0, 0, 0, 0]

    #On prend les samples linéairement pour avoir le même jeu de test pour chaque arbre
    samples = df.sample(samplesize, random_state = 42)
    for i in range(samplesize):
        sample = samples.iloc[i]
        cpred=  inference(node, sample)
        label = int(sample[-1])
        confusionMatrix.iat[label,cpred] += 1
        nbl[label] += 1
    print(confusionMatrix)

    #Calcul des métriques pour chaque classe
    """"
    L’exactitude
    """
    for i in range(4):
        tp = int(confusionMatrix.iat[i,i])
        tpfn = int(confusionMatrix.iat[i,0] + confusionMatrix.iat[i,1] + confusionMatrix.iat[i,2] + confusionMatrix.iat[i,3])
        tpfp = int(confusionMatrix.iat[0,i] + confusionMatrix.iat[1,i] + confusionMatrix.iat[2,i] + confusionMatrix.iat[3,i])
        recall = tp/tpfn if tpfn !=0 else 0
        precision = tp/tpfp if tpfp != 0 else 0
        f1 = 2*(precision*recall)/(precision+recall) if precision != 0 and recall != 0 else 0
        print(i," : recall = {:6.2f}".format(recall), "precision ={:6.2f}".format(precision), "f1 score = {:6.2f}".format(f1))
    
        
    nok = 0
    for i in range (confusionMatrix.columns.size):
        nok += confusionMatrix.iat[i,i]
    print("prédictions correctes : ", nok, "/", samplesize)
    print("prédiction de 0 : {:6.2f}".format(confusionMatrix.iat[0,0]*100/nbl[0]) ,"% (",  confusionMatrix.iat[0,0], "/",nbl[0], ")")
    print("prédiction de 1 : {:6.2f}".format(confusionMatrix.iat[1,1]*100/nbl[1]) ,"% (",  confusionMatrix.iat[1,1], "/",nbl[1], ")")
    print("prédiction de 2 : {:6.2f}".format(confusionMatrix.iat[2,2]*100/nbl[2]) ,"% (",  confusionMatrix.iat[2,2], "/",nbl[2], ")")
    print("prédiction de 3 : {:6.2f}".format(confusionMatrix.iat[3,3]*100/nbl[3]) ,"% (",  confusionMatrix.iat[3,3], "/",nbl[3], ")")
    return 


def main_node():

    df = pd.read_csv("synthetic.csv")

    #On supprime les doublons
    df.drop_duplicates()

    train, test = train_test_split(df, test_size=0.2)

    tree_3 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],3)
    tree_4 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],4)
    tree_5 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],5)
    tree_6 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],6)
    tree_7 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],7)
    tree_8 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],8)

    print("Profondeur = 3")
    #print_tree(tree_3)
    eval_node(tree_3, test, 300)
    print("----------")
    print("----------")
    print("Profondeur = 4")
    #print_tree(tree_4)
    eval_node(tree_4, test, 300)
    print("----------")
    print("----------")
    print("Profondeur = 5")
    #print_tree(tree_5)
    eval_node(tree_5, test, 300)
    print("----------")
    print("----------")
    print("Profondeur = 6")
    #print_tree(tree_6)
    eval_node(tree_6,test, 300)
    print("----------")
    print("----------")
    print("Profondeur = 7")
    #print_tree(tree_7)
    eval_node(tree_7, test, 300)
    print("----------")
    print("----------")
    print("Profondeur = 8")
    #print_tree(tree_8)
    eval_node(tree_8,test, 300)
    print ("----------")
    print("----------")

    #Meilleur modèle : 7 et 6 de profondeur, meme si large marge d'erreur










