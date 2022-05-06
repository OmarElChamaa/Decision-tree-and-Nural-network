#!/usr/bin/env python3
# -*- coding: utf-8 -*_

from pstats import Stats
import pandas as pd
import numpy as np
import scipy.stats as sp
from tpnote import *
from math import isclose

#main_node()
""""
- Arbre de decision • determination d’un meilleur partitionnement
"""

def test_entropie(df):
   print("-----------  TEST ENTROPIE -----------")   
   ok = True
   for i in range (10):
      dftemp = df.sample(200)
      counts = dftemp['Class'].value_counts()
      ent_ref = sp.entropy(counts, base=2)
      ent_test = entropie_df(dftemp)
      print("Entropie with scipy :" , ent_ref)
      print("Notre entropie :", ent_test)
      #On teste le rapprochement des floats avec une tolérance de 10^-10
      if (not isclose(ent_ref, ent_test, rel_tol = 1e-10, abs_tol=0.0)):
         ok = False
         break
   if ok:
      print("**** Entropie ok :) ****")
   else: 
      print("**** Entropie ko :( ****")  
   print("----------- FIN TEST ENTROPIE -----------") 



def test_info_gain():
   # On crée des données fortement dépendante de la longueur et de la largeur
   # On devrait donc avoir un gain pour la longueur et la largeur égaux 
   # à l'entropie et un gain pour le prix 
   print("----------- TEST INFO GAIN -----------")
   data = {
      'Longueur' : [5, 6, 7, 20, 30],
      'Largeur' : [8, 9,  15, 30, 40],
      'Prix' : [100, 250, 600, 200, 500],
      'Class' : ['Mobile','Mobile', 'Mobile', 'Tele', 'Tele']
   } 
   df = pd.DataFrame(data)
   print(df)
   ent = entropie_df(df)
   print("entropie de df: ",ent)

   gainlong, splitlong, partslong = info_gain_quart(df, df.columns.to_list()[0])
   gainlarg, splitlarg, partslarg = info_gain_quart(df, df.columns.to_list()[1])
   gainprix, splitprix, partsprix = info_gain_quart(df, df.columns.to_list()[2])
   print("gain longueur: ",gainlong, ", gain largeur: ", gainlarg, ", gain prix: ", gainprix)
   print("split longueur: ",splitlong,", split largeur: ", splitlarg,", split prix: ", splitprix)
   
   if (isclose(gainlong, ent, rel_tol = 1e-10, abs_tol=0.0) and isclose(gainlarg, ent, rel_tol = 1e-10, abs_tol=0.0) and gainprix < gainlong):
      print("**** Gain OK :) ****")
   print("----------- FIN TEST INFO GAIN -----------")

def test_best_partitions():
   #On teste la fonction super_attribute, qui devrait nous donner comme attribut de split la longueur 
   #et comme partitions les mobile d'un côté et les télé de l'autre
   print("----------- TEST MEILLEUR PARTITION -----------")

   data = {
      'Longueur' : [5, 6, 7, 8, 9, 20, 30, 50],
      'Prix' : [100, 250, 300, 400, 600, 200, 500, 400],
      'Class' : ['Mobile','Mobile', 'Mobile', 'Mobile', 'Mobile', 'Tele', 'Tele', 'Tele']
   } 
   df = pd.DataFrame(data)
   print(df)
   ent = entropie_df(df)
   attribute, gain , split , partitions = super_attribute(df,df.columns.to_list()[:-1]) 
   print("attribut : ", attribute)
   print("gain : ", gain)
   print("split : ", split)
   print("partitions:")
   print(partitions[0])
   print(partitions[1])

   if (isclose(gainlong, ent, rel_tol = 1e-10, abs_tol=0.0)):
      print("**** PARTITION OK :) ****")
   else:
      print("**** PARTITION KO :( ****")

   print("----------- FIN TEST MEILLEUR PARTITION -----------")


def run_tests():  
   df = pd.read_csv("synthetic.csv")
   subdf = df.sample(1000)
   print("Debut des tests")
   test_entropie(subdf)
   test_info_gain()
   test_best_partitions()

run_tests()