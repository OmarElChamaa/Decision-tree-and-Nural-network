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
        
        
        print("shape = ", self.y_train.shape[0])
        print("columns len = ", len(self.y_train.columns))
        self.__weights_initialization(len(self.X_train.columns), self.y_train)

    def __weights_initialization(self, n_attributes, n_classes):
        # X.shape determine nb entrées => tuple => 0 recupere lignes, 1 colonnes
        # 0 car TRANSPOSITION EFFECTUEE AVANT ici (donc ce sera des colonnes)
        # Ft = (self.hidden_layer_sizes[0], X.shape[0]) # First tuple 
        # Lt = (y.shape[0], self.hidden_layer_sizes[-1]) # Last tuple 

        self.weights[0] = np.random.uniform(low=-1.0, high=1.0, size=((self.hidden_layer_sizes[0], n_attributes)))
        self.bias[0] = np.zeros((self.hidden_layer_sizes[0],1)) 
        print("taille weight[0] : ", self.hidden_layer_sizes[0], ",",n_attributes)

        for i in range (1,self.n_layers): # 1 à n-1
            print("taille weight[",i,"] : ", self.hidden_layer_sizes[i], ",",self.hidden_layer_sizes[i-1])
            b_size = (self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1]) 
            self.weights[i] = np.random.uniform(low=-1.0, high=1.0, size=b_size) 
            self.bias[i] = np.zeros(self.hidden_layer_sizes[i]) # Middle
            
            #Lt = (y.shape[0], self.hidden_layer_sizes[-1])
        self.weights[-1] = np.random.uniform(low=-1.0, high=1.0, size=(4,self.hidden_layer_sizes[-1]))
        self.bias[-1] = np.zeros(self.hidden_layer_sizes[-1])
        print("taille sortie",4,self.hidden_layer_sizes[-1])

    # def tanh(Z):
    #     """
    #     Z : non activated outputs
    #     Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
    #     """
    #     A = np.empty(Z.shape)
    #     A = 2.0/(1 + np.exp(-2.0*Z)) - 1 # A = np.tanh(Z)
    #     df = 1-A**2
    #     retur


#act_f = 1 pour sigmoid et 0 pour thanh
    def forward(self,X,y,act_f): # prediction à partir de X, self 
        for i in range(0,len(X)):
            print("i is ", 0)
            trans = X[i:i+1].to_numpy().transpose()
            print("Trans ----------------- au debut \n",trans)
            for j in range(0,self.n_layers+1):
                print("len weights",len(self.weights[j]))
                Z = np.matmul(self.weights[j],trans)
                np.add(Z, self.bias[j])
                print("Z ------------------\n",Z)
                if(j < self.n_layers):
                    print("j is ",j)
                    if(act_f == 1) :
                        #A
                        Z = Utility.sigmoid(Z)
                        print("Z ------------------ after sigmoid\n",Z)
                    else:
                        #A
                        Z = Utility.tanh(Z)
                        print("Z ------------------ after tanh\n",Z)
                    
                    self.error[j] = Z[1]
                    self.activation_matrix[j] = Z[0]
                    print("error ------------------ error\n", self.error[j] ) 
                    trans = Z[0]
                else:

                    Z = Utility.softmax(Z)
                    print("Z ------------------ after softmax\n",Z)
                    #trans = Y
                    self.activation_matrix[j] = Z                   
                    trans = max(Z)
                    self.outputs = Z

            #CE = trans - y[i:i+1].to_numpy().transpose()
            print("Y  -------------------------------- \n",y[i:i+1].to_numpy().transpose())
            print(" Trans ------------------- fin  \n",trans)
            cross_entropy = Utility.cross_entropy_cost(self.activation_matrix[-1],y[i:i+1].to_numpy().transpose())
            print("cross entropy ------------------ ",cross_entropy)
            

#est.forward_prop(X_train,y_train)
def main_nn():
    df = pd.read_csv("synthetic.csv")
    df_columns = df.columns.values.tolist()
    attributes = df_columns[:-1]
    label = df_columns[-1:]
    print(attributes)
    print("labe",label)

    X = df[attributes]
    y = df[label]
    print("Y is before dummy  ",y)
    yd = pd.get_dummies(y.astype(str))
    print("Y after dummy is ",yd)
    yd.head()

    print("X is ",X)
    print("Y is ",yd)

    X_train, X_test, y_train, y_test = train_test_split(X, yd, test_size=0.15)
    print("ytrain : ", y_train)
    test = NeuralNet(X_train, y_train, X_test, y_test,hidden_layer_sizes=(10,8,6))
    df = pd.read_csv("./predictions/y_pred_NN_relu_10-8-6.csv")
    # Z = Utility.softmax(df)
    # print("softmax \n",Z)
    test.forward(X_train,y_train,1)
    print("test forward -------------------------- error, ",test.error)
    print("activation ------------------------------ ",test.activation_matrix)
    print("Outputs ================================== ", test.outputs)



main_nn()