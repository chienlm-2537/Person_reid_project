import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
import pandas as pd

# tao lop mang 
class NeuralNet():
    '''
    A two layer neural network
    '''
        
    def __init__(self, layers=[8,5,1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        # self.W1 = 0.5
        # self.W2 = 0.5
        # self.b1 = 0
        # self.b2 = 0
        np.random.seed(1) # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) # mang trong so W1 co kich thuoc 8*5
        self.params['b1']  =np.random.randn(self.layers[1],) 
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2],)
        # print(self.params["W2"])
        # print("thudong")
        # print(self.params['b2'])
        # print(self.params["W2"].dtype)
        print(self.params["W1"])
    #ham kich hoat
    def relu(self,Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less 
        than zero are set to zero.
        '''
        return np.maximum(0,Z)
    def sigmoid(self,Z):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1.0/(1.0+np.exp(-Z))
    #ham mat mat
    def entropy_loss(self,y, yhat):
        nsample = len(y)
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
        return loss

    #lan truyen tien
    def forward_propagation(self):
            '''
            Performs the forward propagation
            '''
            
            Z1 = self.X.dot(self.params['W1']) + self.params['b1']
            # print(self.X)
            # print(self.X.dtype)
            A1 = self.relu(Z1)
            Z2 = A1.dot(self.params['W2']) + self.params['b2']
            # print(Z2)
            yhat = self.sigmoid(Z2)
            # print(yhat)
            loss = self.entropy_loss(self.y,yhat)
            # print("ttttt")
            # print(loss)
            # save calculated parameters     
            self.params['Z1'] = Z1
            self.params['Z2'] = Z2
            self.params['A1'] = A1

            return yhat,loss
    #lan truyen nguoc
    def back_propagation(self,yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        def dRelu(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x
        
        dl_wrt_yhat = -(np.divide(self.y,yhat) - np.divide((1 - self.y),(1-yhat)))
        dl_wrt_sig = yhat * (1-yhat)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

       #update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    # dau vao X va nhan y, ham lap di lap lai qua trinh chuyen tiep va chuyen nguoc voi so lan lap lai duoc chi dinh
    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights() #initialize weights and bias


        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)
            
        print(self.params["W1"])   
    def predict(self, X):
        '''
        Predicts on a test data
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)              

                                
    def acc(self, y, yhat):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc


    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()

with open('pima_indians_diabetes.csv', 'r') as file:
    reader = csv.reader(file)
    
    y = []
    X = []
    for col in reader:
        # print(col)
        y.append(col[8])
        out = np.array(y, dtype='int64') 
        out = np.reshape(out,(-1,1))   
        for i  in range(0,8): 
            X.append(col[i])
        X_np = np.array(X, dtype='float64') 
        In = np.reshape(X_np,(-1,8))   
    # print(In.shape)
    # print(len(y))
    Xtrain, Xtest, ytrain, ytest = train_test_split(In, out, test_size=0.3, random_state=42)
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)   
# print(Xtrain.shape)
# print(Xtrain.dtype)
# print(Xtest.shape)
# print(ytrain.shape)
# print(ytrain.dtype)
# print(ytest.shape)

nn = NeuralNet(layers=[8,5,1], learning_rate=0.001, iterations=100) # create the NN model
nn.fit(Xtrain, ytrain) #train the model
# nn.plot_loss()
yhat = nn.predict(Xtrain)
# print(yhat.shape)
acc = nn.acc(ytrain,yhat)
print(acc)



# add header names
# headers =  ['age', 'sex','chest_pain','resting_blood_pressure',  
#         'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
#         'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
#         'num_of_major_vessels','thal', 'heart_disease']

# heart_df = pd.read_csv('heart.dat', sep=' ', names=headers)
# #convert imput to numpy arrays
# X = heart_df.drop(columns=['heart_disease'])

# #replace target class with 0 and 1 
# #1 means "have heart disease" and 0 means "do not have heart disease"
# heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
# heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)

# y_label = heart_df['heart_disease'].values.reshape(X.shape[0], 1)

# #split data into train and test set
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

# #standardize the dataset
# sc = StandardScaler()
# sc.fit(Xtrain)
# Xtrain = sc.transform(Xtrain)
# Xtest = sc.transform(Xtest)

# print(Xtrain.shape)
# print(Xtrain.dtype)
# print(Xtest.shape)
# print(ytrain.shape)
# print(ytrain.dtype)
# print(ytest.shape)
# nn = NeuralNet(layers=[13,8,1], learning_rate=0.001, iterations=100) # create the NN model
# nn.fit(Xtrain, ytrain) #train the model
# nn.plot_loss()