#!/usr/bin/env python3
# -*- coding: utf-8 -*_

import pandas as pd
import numpy as np
from tpnote import *



def test_info_gain():
    # On crée des données fortement dépendante de la longueur et de la largeur
    # On devrait donc avoir un gain pour la longueur et la largeur similaires
    # et proche de 1, comme l'entropie sera inférieure à 1 (deux classes)
    data = {
        'Longueur' : [5, 6, 7, 20, 30],
        'Largeur' : [8, 9,  15, 30, 40],
        'Prix' : [100, 250, 600, 200, 500],
        'Class' : ['Mobile','Mobile', 'Mobile', 'Tele', 'Tele']
    } 
    df = pd.DataFrame(data)
    print(df)
    print("entropie de df: ",entropie_df(df))
    print("entropie long: ", entropie_df(df.columns.to_list()[0]))
    print("entropie larg: ", entropie_df(df.columns.to_list()[1]))
    print("entropie prix: ", entropie_df(df.columns.to_list()[2]))


    gainlong, splitlong, partslong = info_gain(df, df.columns.to_list()[0])
    gainlarg, splitlarg, partslarg = info_gain(df, df.columns.to_list()[1])
    gainprix, splitprix, partsprix = info_gain(df, df.columns.to_list()[2])
    print("gain longueur: ",gainlong, ", gain largeur: ", gainlarg, ", gain prix: ", gainprix)
    print("split longueur: ",splitlong,", split largeur: ", splitlarg,", split prix: ", splitprix)


test_info_gain()
""""
- Arbre de decision • calcul de l’entropie d’une partition
- Arbre de decision • calcul du gain d’une partition
- Arbre de decision • determination d’un meilleur partitionnement
"""