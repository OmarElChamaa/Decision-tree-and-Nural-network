#!/usr/bin/env python3
# -*- coding: utf-8 -*_

from pstats import Stats
import pandas as pd
import numpy as np
import scipy.stats as sp
from tpnote import *

df = pd.read_csv("synthetic.csv")

def test_entropie():
    counts = df['Class'].value_counts()
    print("Entropie with scipy " , sp.entropy(counts))
    print("Notre entropie", entropie_df(df))
    print(":D")    

test_entropie()
# def test_best_partitions(df):
    

    
# test_info_gain()
""""
- Arbre de decision • calcul de l’entropie d’une partition
- Arbre de decision • calcul du gain d’une partition
- Arbre de decision • determination d’un meilleur partitionnement
"""