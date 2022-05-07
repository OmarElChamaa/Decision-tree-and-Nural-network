import numpy as np
import pandas as pd
from utility import Utility
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, X_train, y_train, X_test = None, y_test = None,
    hidden_layer_sizes  = (10,8,6), activation='identity', learning_rate=0.1, epoch=200):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_layers = len(hidden_layer_sizes)
        self.weights = [None] * (self.n_layers + 1)
        self.weightedInput = [None] * (self.n_layers + 1) # Z
        self.bias = [None] * (self.n_layers + 1)
        self.outputs = [None]
        
        self.error = [None] * (self.n_layers + 1)

        self.activation = activation # (****) REMPLACER PAR :
        #getattr(object_name, attribute_name[, default_value])
        # un nombre d’epoch pendant lequel entrainer le modèle

        
        self.learning_rate = learning_rate

        self.curr_epoch = 0
        self.epoch = epoch

        # une liste de matrices d’activations
        self.activation_matrix = [None] * (self.n_layers + 1) # A
        
        
        self.__weights_initialization(len(self.X_train.columns), self.y_train)

    def __weights_initialization(self, n_attributes, n_classes):
        self.weights[0] = np.random.uniform(low=-1.0, high=1.0, size=((self.hidden_layer_sizes[0], n_attributes)))
        self.bias[0] = np.zeros((self.hidden_layer_sizes[0],1)) 

        for i in range (1,self.n_layers): # 1 à n-1
            b_size = (self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1]) 
            self.weights[i] = np.random.uniform(low=-1.0, high=1.0, size=b_size) 
            self.bias[i] = np.zeros(self.hidden_layer_sizes[i]) # Middle
            
        self.weights[-1] = np.random.uniform(low=-1.0, high=1.0, size=(4,self.hidden_layer_sizes[-1]))
        self.bias[-1] = np.zeros(self.hidden_layer_sizes[-1])
        

    def backward(self,X,y) : 
        for j in range(0,len(X)):

            delta = [None] * (self.n_layers + 1)
            dW = [None] * (self.n_layers + 1)
            db = [None] * (self.n_layers + 1)

            delta[-1] = self.activation_matrix[-1] - y[j:j+1].transpose()
            dW[-1] = np.dot(delta[-1], self.activation_matrix[-2].transpose())
            db[-1] = delta[-1]

            for i in range(self.n_layers-1,-1,-1) :
                weights = self.weights
                delta[i] = np.multiply((weights[i+1].transpose() @ delta[i+1]),self.error[i]) 

                if(i == 0) :  # A[l-1] correspond aux entrées du réseau
                    dW[0] = delta[0] @ X[j:j+1]
                else:
                    dW[i] = delta[i] @ self.activation_matrix[i-1].transpose()
                db[i] = delta[i]

            for i in range(0,self.n_layers):
                self.weights[i] = self.weights[i] - self.learning_rate*dW[i]
                self.bias[i] = self.bias[i] - (self.learning_rate*db[i]).values.tolist()

        self.weights[0].transpose

    #act_f = 1 pour sigmoid et 0 pour thanh
    def forward(self,X,y,act_f): # prediction à partir de X, self 
        for i in range(0,len(X)):

            trans = X[i:i+1].to_numpy().transpose()

            for j in range(0,self.n_layers+1):

                Z = np.matmul(self.weights[j],trans)
                np.add(Z, self.bias[j])

                if(j < self.n_layers):
                    if(act_f == 1) :
                        #A
                        Z = Utility.sigmoid(Z)
                    else:
                        #A
                        Z = Utility.tanh(Z)
                    
                    self.error[j] = Z[1]
                    self.activation_matrix[j] = Z[0]
                    trans = Z[0]

                else:

                    Z = Utility.softmax(Z)
                    self.activation_matrix[j] = Z   
                    #prédiction                
                    trans = max(Z)
                    self.outputs = Z     

            cross_entropy = Utility.cross_entropy_cost(self.activation_matrix[-1],y[i:i+1].to_numpy().transpose())

            #passe arrière
            #self.backward(X, y)





