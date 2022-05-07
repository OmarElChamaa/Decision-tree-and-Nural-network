#!/usr/bin/env python3
# -*- coding: utf-8 -*_

from pstats import Stats
import pandas as pd
import numpy as np
import scipy.stats as sp
from arbre import *
from neurones import *
from math import isclose

#Fonction de test de l'entropie
def test_entropie(df):
   print("-----------  TEST ENTROPIE -----------")   
   ok = True
   for i in range (10):
      print("-- Test entropie ", i, " --")
      dftemp = df.sample(200)
      counts = dftemp['Class'].value_counts()
      ent_ref = sp.entropy(counts, base=2)
      ent_test = entropie_df(dftemp)
      print("Entropie with scipy :" , ent_ref)
      print("Notre entropie :", ent_test)
      #On teste le rapprochement des floats avec une tolérance de 10^-10
      if (not isclose(ent_ref, ent_test, rel_tol = 1e-10, abs_tol=0.0)):
         ok = False
         print("entropie de référence et de test différentes")
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
   #et comme partitions les mobile d'un côté et les télé de l'autre, en choisissant un quartile à 0.75
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
   print("quartiles :")
   print("0.25 : ", df.quantile(0.25, interpolation='nearest'))
   print("0.50 : ", df.quantile(0.50, interpolation='nearest'))
   print("0.75 : ", df.quantile(0.75, interpolation='nearest'))
   print(partitions[0])
   print(partitions[1])

   if (attribute == 'Longueur' and isclose(gain, ent, rel_tol = 1e-10, abs_tol=0.0) and split == df['Longueur'].quantile(0.75, interpolation = 'nearest')):
      print("**** PARTITION OK :) ****")
   else:
      print("**** PARTITION KO :( ****")

   print("----------- FIN TEST MEILLEUR PARTITION -----------")

def test_propagation(df):
   print("----------- TEST PASSE AVANT -----------")
   df_columns = df.columns.values.tolist()
   attributes = df_columns[:-1]
   label = df_columns[-1:]

   X = df[attributes]
   y = df[label]

   yd = pd.get_dummies(y.astype(str))

   X_train, X_test, y_train, y_test = train_test_split(X, yd, test_size=0.20)
   test1 = NeuralNet(X_train, y_train, X_test, y_test,hidden_layer_sizes=(10,8,6))
   test2 = NeuralNet(X_train, y_train, X_test, y_test,hidden_layer_sizes=(10,8,4))
   test3 = NeuralNet(X_train, y_train, X_test, y_test,hidden_layer_sizes=(6,4))
   test1.forward(X_train,y_train,1)
   test2.forward(X_train,y_train,0)
   test3.forward(X_train,y_train,0)

   #On vérifie que la sortie correspond bien à un softmax
   sumsoft1 = test1.outputs[0][0] + test1.outputs[1][0] + test1.outputs[2][0] + test1.outputs[3][0] 
   sumsoft2 = test2.outputs[0][0] + test2.outputs[1][0] + test2.outputs[2][0] + test2.outputs[3][0]
   sumsoft3 = test3.outputs[0][0] + test3.outputs[1][0] + test3.outputs[2][0] + test3.outputs[3][0]
   print("sumsoft 1 ", sumsoft1)
   print("sumsoft 2 ", sumsoft2)
   print("sumsoft 3 ", sumsoft3)
   print("les longueurs des activations de 1 : ", len(test1.activation_matrix[0]), ", ", len(test1.activation_matrix[1])
   , ", ", len(test1.activation_matrix[2]), ", ", len(test1.activation_matrix[3]))
   print("les longueurs des activations de 2 : ", len(test2.activation_matrix[0]), ", ", len(test2.activation_matrix[1])
   , ", ", len(test2.activation_matrix[2]), ", ", len(test2.activation_matrix[3]))
   print("les longueurs des activations de 3 : ", len(test3.activation_matrix[0]), ", ", len(test3.activation_matrix[1])
   , ", ", len(test3.activation_matrix[2]))
   #Vérification de la taille des matrices d'activations
   size1_ok = (len(test1.activation_matrix[0]) == 10 and len(test1.activation_matrix[1]) == 8 and len(
      test1.activation_matrix[2]) == 6 and len(test1.activation_matrix[3]) == 4)
   size2_ok = (len(test2.activation_matrix[0]) == 10 and len(test2.activation_matrix[1]) == 8 and len(
      test2.activation_matrix[2]) == 4 and len(test2.activation_matrix[3]) == 4)  
   size3_ok = (len(test3.activation_matrix[0]) == 6 and len(test3.activation_matrix[1]) == 4 and len(
      test3.activation_matrix[2]) == 4) 
   sizes_ok = size1_ok and size2_ok and size3_ok
   if isclose(sumsoft1, 1, rel_tol = 1e-10, abs_tol=0.0) and isclose(sumsoft2, 1, rel_tol = 1e-10, abs_tol=0.0) and isclose(
      sumsoft3, 1, rel_tol = 1e-10, abs_tol=0.0) and sizes_ok:
      print("**** PASSE AVANT OK :) ****")
   else:
      print("**** PASSE AVANT KO :( ****")
   print("----------- FIN TEST PASSE AVANT -----------")


def eval_model(node, nn, df, samplesize):
   if (node is None):
      eval_nn(nn, df, samplesize)
   else:
      eval_node(node, df, samplesize)

#Procédure permettant d'évaluer le modèle nn sur un partie de df
#On crée la matrice de confusion du modèle
def eval_nn(nn, df, samplesize):
   return

#Procédure permettant d'évaluer l'arbre de décision node sur un partie de df
#On crée la matrice de confusion du modèle et on calcule les métriques
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

   #On prend les même samples pour avoir le même jeu de test pour chaque arbre
   samples = df.sample(samplesize, random_state = 42)
   for i in range(samplesize):
      sample = samples.iloc[i]
      cpred=  int(inference(node, sample))
      label = int(sample[-1])
      confusionMatrix.iat[label,cpred] += 1
      nbl[label] += 1
   print(confusionMatrix)

   #Calcul des métriques pour chaque classe
   for i in range(4):
      tp = int(confusionMatrix.iat[i,i])
      tpfn = int(confusionMatrix.iat[i,0] + confusionMatrix.iat[i,1] + confusionMatrix.iat[i,2] + confusionMatrix.iat[i,3])
      tpfp = int(confusionMatrix.iat[0,i] + confusionMatrix.iat[1,i] + confusionMatrix.iat[2,i] + confusionMatrix.iat[3,i])
      recall = tp/tpfn if tpfn !=0 else 0
      precision = tp/tpfp if tpfp != 0 else 0
      f1 = 2*(precision*recall)/(precision+recall) if precision != 0 and recall != 0 else 0
      accurracy = tp/samplesize
      print(i," : accuracy = {:6.2f}".format(accurracy),", recall = {:6.2f}".format(recall), "precision ={:6.2f}".format(precision), "f1 score = {:6.2f}".format(f1))
        
   nok = 0
   for i in range (confusionMatrix.columns.size):
      nok += confusionMatrix.iat[i,i]
   print("prédictions ok : {:6.2f}".format(nok*100/samplesize),"(", nok, "/", samplesize,")")
   print("prédiction de 0 : {:6.2f}".format(confusionMatrix.iat[0,0]*100/nbl[0]) ,"% (",  confusionMatrix.iat[0,0], "/",nbl[0], ")")
   print("prédiction de 1 : {:6.2f}".format(confusionMatrix.iat[1,1]*100/nbl[1]) ,"% (",  confusionMatrix.iat[1,1], "/",nbl[1], ")")
   print("prédiction de 2 : {:6.2f}".format(confusionMatrix.iat[2,2]*100/nbl[2]) ,"% (",  confusionMatrix.iat[2,2], "/",nbl[2], ")")
   print("prédiction de 3 : {:6.2f}".format(confusionMatrix.iat[3,3]*100/nbl[3]) ,"% (",  confusionMatrix.iat[3,3], "/",nbl[3], ")")
   return 


#Fonction permettant de créer des arbres de décision de profondeur 3 à 8 et de les évaluer
def main_node():

   df = pd.read_csv("synthetic.csv")

   #On supprime les doublons
   df = df.drop_duplicates()

   #normalisation des données
   normalized_df=(df-df.mean())/df.std()
   normalized_df['Class'] = df['Class']

   # on supprime les outliers via le calcul du zscore 
   df = df[(np.abs(stats.zscore(df)) < 2.9).all(axis=1)]

   train, test = train_test_split(normalized_df, test_size=0.2)

   tree_3 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],3)
   tree_4 = ze_tree(train,0,'Class',train.columns.tolist()[:-1],4)
   tree_5 = ze_tree(train,0,'Class',train.columns.tolist()[:-1],5)
   tree_6 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],6)
   tree_7 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],7)
   tree_8 = ze_tree(train,0,'Class',df.columns.tolist()[:-1],8)

   print("Profondeur = 3")
   #print_tree(tree_3)
   eval_model(tree_3, None, test, 300)
   print("----------")
   print("----------")
   print("Profondeur = 4")
   #print_tree(tree_4)
   eval_model(tree_4, None, test, 300)
   print("----------")
   print("----------")
   print("Profondeur = 5")
   #print_tree(tree_5)
   eval_model(tree_5, None, test, 300)
   print("----------")
   print("----------")
   print("Profondeur = 6")
   #print_tree(tree_6)
   eval_model(tree_6, None, test, 300)
   print("----------")
   print("----------")
   print("Profondeur = 7")
   #print_tree(tree_7)
   eval_model(tree_7, None, test, 300)
   print("----------")
   print("----------")
   print("Profondeur = 8")
   #print_tree(tree_8)
   eval_model(tree_8, None, test, 300)
   print("----------")
   print("----------")

#est.forward_prop(X_train,y_train)
def main_nn():
   df = pd.read_csv("synthetic.csv")
   df_columns = df.columns.values.tolist()
   attributes = df_columns[:-1]
   label = df_columns[-1:]

   X = df[attributes]
   y = df[label]

   yd = pd.get_dummies(y.astype(str))

   yd.head()


   X_train, X_test, y_train, y_test = train_test_split(X, yd, test_size=0.15)

   test = NeuralNet(X_train, y_train, X_test, y_test,hidden_layer_sizes=(10,8,6))
   test.forward(X_train,y_train,1)




#Fonction permettant de créer des arbres de décision de profondeur 3 à 8 et de les évaluer
def reseauTest(str):
   df_anne_pred = pd.read_csv(str)
   size = df_anne_pred.shape[0]
   df_y_test = pd.read_csv("./predictions/y_test.csv")
   predmat = {
            #0  1  2  3
      '0' : [0, 0, 0, 0],
      '1' : [0, 0, 0, 0],
      '2' : [0, 0, 0, 0],
      '3' : [0, 0, 0, 0]
   } 
   nbl = [0, 0, 0, 0]   

   confusionMatrix = pd.DataFrame(predmat)
   for index,row in df_anne_pred.iterrows() : 
      max_val = 0 
      max_index = 0 
      for j in range(4):
         if(row[j]>max_val) :
            max_val = row[j]
            max_index = j 

      cpred =  int(max_index)
      label = int(df_y_test.loc[index])
      confusionMatrix.iat[label,cpred] += 1
      nbl[label] += 1   
   print(confusionMatrix) 
   
   #Calcul des métriques pour chaque classe
   for i in range(4):
      tp = int(confusionMatrix.iat[i,i])
      tpfn = int(confusionMatrix.iat[i,0] + confusionMatrix.iat[i,1] + confusionMatrix.iat[i,2] + confusionMatrix.iat[i,3])
      tpfp = int(confusionMatrix.iat[0,i] + confusionMatrix.iat[1,i] + confusionMatrix.iat[2,i] + confusionMatrix.iat[3,i])
      recall = tp/tpfn if tpfn !=0 else 0
      precision = tp/tpfp if tpfp != 0 else 0
      f1 = 2*(precision*recall)/(precision+recall) if precision != 0 and recall != 0 else 0
      accurracy = tp/size
      print(i," : accuracy = {:6.2f}".format(accurracy),", recall = {:6.2f}".format(recall), "precision ={:6.2f}".format(precision), "f1 score = {:6.2f}".format(f1))
        
   nok = 0
   for i in range (confusionMatrix.columns.size):
      nok += confusionMatrix.iat[i,i]
   print("prédictions ok : {:6.2f}".format(nok*100/size),"(", nok, "/", size,")")
   print("prédiction de 0 : {:6.2f}".format(confusionMatrix.iat[0,0]*100/nbl[0]) ,"% (",  confusionMatrix.iat[0,0], "/",nbl[0], ")")
   print("prédiction de 1 : {:6.2f}".format(confusionMatrix.iat[1,1]*100/nbl[1]) ,"% (",  confusionMatrix.iat[1,1], "/",nbl[1], ")")
   print("prédiction de 2 : {:6.2f}".format(confusionMatrix.iat[2,2]*100/nbl[2]) ,"% (",  confusionMatrix.iat[2,2], "/",nbl[2], ")")
   print("prédiction de 3 : {:6.2f}".format(confusionMatrix.iat[3,3]*100/nbl[3]) ,"% (",  confusionMatrix.iat[3,3], "/",nbl[3], ")")
   return 



def run_tests():  
   df = pd.read_csv("synthetic.csv")
   subdf = df.sample(1000)
   print("Debut des tests")
   test_entropie(subdf)
   test_info_gain()
   test_best_partitions()
   test_propagation(df)
   print("test resultats atanh_10-8-6")  
   reseauTest("./predictions/y_pred_NN_tanh_10-8-6.csv")

   print("test resultats atanh_10-8-4")  
   reseauTest("./predictions/y_pred_NN_tanh_10-8-4.csv")

   print("test resultats atanh_6-4")  
   reseauTest("./predictions/y_pred_NN_tanh_6-4.csv")

   print("test resultats relu 10-8-4")  
   reseauTest("./predictions/y_pred_NN_relu_10-8-4.csv")

   print("test resultats relu 10-8-6")  
   reseauTest("./predictions/y_pred_NN_relu_10-8-6.csv")

   print("test resultats relu 6-4")  
   reseauTest("./predictions/y_pred_NN_relu_6-4.csv")
   
   
   

run_tests()
#pour créer les arbres, d'une hauteur de 3 à 8 et les évaluer
#main_node()
#main_nn()