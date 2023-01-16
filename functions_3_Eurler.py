#from Project_2_dataExtraction import load_mnist
import numpy as np
import os
import cvxopt as cvx
import gzip
from time import time
seed = 1985602


def load_mnist(path, kind='train'):
    #Author: Corrado
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels







def get_data():
    '''
        #####################################################
            Function to load data and return the data
        #####################################################
        It returns the X of Labels 1 and 5
    '''
    cwd = os.getcwd() #Current Directory path
    path_parent = os.path.dirname(cwd) #Getting parent directory path, because the data is in that directory
    X_all_labels, y_all_labels = load_mnist(path_parent, kind='train')

    """
    We are only interested in the items with label 1, 5 and 7.
    Only a subset of 1000 samples per class will be used.
    """
    indexLabel1 = np.where((y_all_labels==1))
    xLabel1 =  X_all_labels[indexLabel1][:1000,:].astype('float64')
    yLabel1 = y_all_labels[indexLabel1][:1000].astype('float64')

    indexLabel5 = np.where((y_all_labels==5))
    xLabel5 =  X_all_labels[indexLabel5][:1000,:].astype('float64')
    yLabel5 = y_all_labels[indexLabel5][:1000].astype('float64')

    indexLabel7 = np.where((y_all_labels==7))
    xLabel7 =  X_all_labels[indexLabel7][:1000,:].astype('float64')
    yLabel7 = y_all_labels[indexLabel7][:1000].astype('float64')
    return xLabel1,xLabel5





def PreProcessData(x):
    '''
        #####################################################
            Scaling features and convert labels to 1,-1
        #####################################################
        X as input to be scaled and creating vectors of labels
        This function return features and labels
    '''
    #Converting features data np array and scale them by dividing by 255
    label1_data = np.array(x[0]) / 255.0 
    label5_data = np.array(x[1]) / 255.0 

    #Creating two different arrays with different values as  our different target classes
    y1 = np.ones((1000,1)) #Target class is sitll 1
    y5 = np.full((1000,1),-1) #Renaming target class 5 as -1

    #Concatenating these two X arrays to one features array as X
    X = np.concatenate([label1_data,label5_data]) #Features array
    Y = np.concatenate([y1,y5]) #And our Target array
    return X,Y

def CreateTrainTestSplit(X,Y):
    '''
        #####################################################
            Splitting the data to training and test sets
        #####################################################
        X,Y as inputs to be shuffled randomly with fixed seed,
        and then splitted 80%, 20% for train set and test set
    '''
    trainset_size = int(len(X) * 0.8) #Train set size
    Xy = np.concatenate((X,Y),axis=1) #Concatenating X and Y vectors
    np.random.seed(seed) #Setting the seed for numpy to have reproducible data sets
    np.random.shuffle(Xy) #Shuffling the whole dataset
    train_dataset = Xy[:trainset_size] #Training dataset
    test_dataset = Xy[trainset_size:] #Testing dataset
    trainX,trainY = train_dataset[:,:-1],train_dataset[:,-1] #X,Y of train dataset (Features and Labels)
    testX,testY = test_dataset[:,:-1],test_dataset[:,-1] #X,Y of test dataset (Features and Labels)
    
    #Reshaping both labels arrays to have 2 dimentions
    trainY = trainY.reshape(-1,1)  
    testY = testY.reshape(-1,1) 
    return trainX,testX,trainY,testY

def KFoldsSplit(trainX,trainY,kfolds=5):
    '''
        #####################################################
            Creating k-folds from training data set
        #####################################################
        trainX,trainY as inputs to create K folds and return folds with data
    '''
    trainset = np.hstack((trainX,trainY)) #Concatenating both features and labels
    folds = [] #To store our folds
    folds_length = int(len(trainX)/kfolds) #Obtaining each fold size
    #Creating an array with eqaul values of folds_length and then muliplying them by 0 to K to create an array of indexes
    fold_index = np.full(kfolds+1,dtype=np.int32,fill_value=folds_length) * np.arange(kfolds+1)

    #Using enumerate to access to the index of each element in fold_size array
    for i in range(len(fold_index)-1):
        begin = fold_index[i] #Start index
        end = fold_index[i+1] #End index
        fold = trainset[begin:end] #Creating fold
        folds.append(fold) #Appending to the final list

    return np.array(folds) 



def Accuracy(ypred,ytrue):
    '''
        #####################################################
                    Getting accuracy of the model
        #####################################################
        predicted values and ground truth labels as inputs 
        and return accuracy as percentage
    '''
    return np.count_nonzero(ypred.ravel()==ytrue.ravel())/len(ypred)


def ConfusionMatrix(ypred, ytrue):
    '''
        #####################################################
                        Confusion Matrix
        #####################################################
        predicted values and ground truth labels as inputs 
        and return Confusion Matrix
    '''
    cfm = np.zeros((2, 2),dtype=int) #An array of zeros with 2,2 dim with data type of integer
    #Removing 2nd dim from the arrays
    yt = ytrue.ravel() 
    yp = ypred.ravel()

    #Filling the confusion Matrix
    cfm[0][0] = np.count_nonzero((yt==1) & (yt==yp)) #TrueValue = 1 and true prediction     TruePositive
    cfm[0][1] = np.count_nonzero((yt==1) & (yt!=yp)) #TrueValue = 1 and false prediction    FalsePositive
    cfm[1][0] = np.count_nonzero((yt==-1)&(yt!=yp)) #TrueValue = -1 and false predicion     FalseNegative
    cfm[1][1] = np.count_nonzero((yt==-1)&(yt==yp)) #TrueValue = -1 and true prediction     TrueNegative
    return cfm


class MVPSVM:
    '''
        ###################################################################
            Soft margin MVP-SVM, solving with CVXOPT library
        ###################################################################
        Parameters to input: C,Lambda and q
        C and Lambda are already obtained by hyperparamter tuning using K-fold cross validation in previous question in homework
        C = 0.00001
        Lambda = 2.4
        Kernel: Polynomial
    '''
    
    def __init__(self,C=0.1,lambda_param=2.4,kernel="poly"):
        self.C = C #C parameter
        self.lambda_param = lambda_param #Lambda param of polynomial kernel
        self.q = 2 #q value is set to 2 based on question asked in the homework
        self.intercept = 0 #Intercept inited with 0
        self.X = None #Features
        self.Y = None #Labels
        self.samples = 0 #No. of samples
        self.solve_time = 0 #Optimization time
        self.n_sv = None #Number of support vectors
        if kernel == "poly":
            self.kernel = lambda x1,x2: np.power(np.dot(x1, x2.T) + 1, self.lambda_param)
        elif kernel == "rbf":
            self.kernel = lambda x1,x2: self.gaussian_kernel(x1,x2)
        #------specific variables to SVM using MVP method------
        self.alpha = None #Alpha vector
        self.K = None
        self.Q = None
        self.eps = 1e-6
        self.n_iterations = 0 #Number of iterations
        self.n_fev = 0 #Number of function evalution

        
        
    def gaussian_kernel(self,x1,x2):
        a1 = np.sum(x1**2,axis=1).reshape(-1,1)
        a2 = np.sum(x2**2,axis=1)
        return np.exp(-1*self.lambda_param * (a1+a2 - 2*(x1.dot(x2.T))))
    
    def objective_function(self,Q):
        first_part = 0.5 * (self.alpha.T.dot(Q)).dot(self.alpha)
        #second_part = np.exp.T.dot(self.alpha)
        e = np.ones_like(self.alpha)
        second_part = e.T.dot(self.alpha)
        #print(second_part)
        return first_part - second_part

    
    def calculate_beta_maximum(self,d1,d2,alpha):
        beta = 0

        if d1> 0 and d2 > 0:
            beta = min(self.C-alpha[0],self.C-alpha[1])
        elif d1> 0 and d2<0:
            beta = min(self.C-alpha[0],alpha[1])
        elif d1 < 0 and d2 > 0:
            beta = min(alpha[0],self.C-alpha[1])
        else:
            beta = min(alpha[0],alpha[1])

        return beta
        


    
    def calculate_working_set(self):
        '''
        #####################################################
                        Calculating working set
        #####################################################
        Calculating working set and returns
        optimal_solutions: boolean to check if it finds optimal solution or not
        Working_set
        Working_set_not
        '''
        optimal_solution = False #Initilizing this boolean for KKT condition check
        self.Y = self.Y.ravel() #Flattening the Y array
        grady = self.Q.dot(self.alpha) - 1 #Gradient of Y
        neg_grady= -1 * grady/self.Y #Negative gradient divided by Y(label data)

        #Calculating R and S sets
        R = np.where((self.alpha < self.eps) & (self.Y == +1) | (self.alpha > self.C-self.eps) & (self.Y == -1) | (self.alpha > self.eps) & (self.alpha < self.C-self.eps))[0]
        S = np.where((self.alpha < self.eps) & (self.Y == -1) | (self.alpha > self.C-self.eps) & (self.Y == +1) | (self.alpha > self.eps) & (self.alpha < self.C-self.eps))[0]

        #Using dictionary object to make it pair-wise 
        gradient_dict = {}
        for i in range(len(neg_grady)):
            gradient_dict[i] = neg_grady[i]

        
        R_dict = dict((k, gradient_dict[k]) for k in R) #R set dictionary
        sorted_R = {k: v for k, v in sorted(R_dict.items(), key=lambda item: item[1])} #Sorting R set-dictonary
        I = list(sorted_R.keys())[-1]
        
        S_dict = dict((k, gradient_dict[k]) for k in S)  #S set dictonary
        sorted_S = {k: v for k, v in sorted(S_dict.items(), key=lambda item: item[1])} #Sorting S set-dictionary
        J = list(sorted_S.keys())[0]
        
        Working_set = [I,J] #Working set by adding  R and S sets

        m = max(neg_grady[R]) #Small m
        M = min(neg_grady[S]) #Big M

        d1 = self.Y[I]
        d2 = -1*self.Y[J]
        #Optimality checking by checking KKT conditions
        if m-M < 1e-3:
            optimal_solution = True
            self.kkt_mM_diff = m-M
            self.m = m
            self.M = M
        
        newQ = self.Q[np.ix_(Working_set,Working_set)] #New hessian matrix calculated from working set
   
            
        return optimal_solution,Working_set, newQ,grady,d1,d2



    def fit(self,X,Y):
        '''
        #####################################################
                            Main function
        #####################################################
        Creating necessary parameters for the solver and running it
        '''
        
        self.X = X #Assigning our features
        self.Y = Y #Assigning our labels
        self.samples = len(self.X) #Getting the count of samples
        eps = 1e-6 #Tolerance

        #Initilizing K and Q matrixes
        self.K = self.kernel(self.X,self.X) #Kernel Matrix
        self.Q = np.outer(self.Y,self.Y) * self.K #Hessian matrix
       

        #Initilizing alpha
        self.alpha = np.zeros(self.samples)
        start_time = time()
        for i in range(10000):
            optimal_solution,Working_set, newQ,grady,d1,d2 = self.calculate_working_set()
        
            if not optimal_solution:
                #d,d* and beta*
                d = np.array([d1,d2]).reshape(-1,1)
                d_star = np.zeros(2)
                wd = grady[Working_set].dot(d)
                if wd !=0 :
                    if wd < 0:
                        d_star = d
                    else:
                        d_star = -d
                    beta_bar = self.calculate_beta_maximum(d_star[0],d_star[1],self.alpha[Working_set])
                    if beta_bar == 0:
                        beta_star = 0
                    elif d_star.T.dot(newQ).dot(d_star) == 0:
                        beta_star == beta_bar
                    elif d_star.T.dot(newQ).dot(d_star) > 0:
                        beta_nv = (-grady[Working_set].dot(d_star))/(d_star.T.dot(newQ).dot(d_star))
                        beta_star = min(beta_bar, beta_nv)
                    else:
                        pass

            else:
                break
            
            self.n_iterations += 1 #Adding to total algorithm iteration
            alpha_star = self.alpha[Working_set] + beta_star * d_star.T #Updating alpha with new beta* and d*
            self.alpha[Working_set] = alpha_star
        self.solve_time = time() - start_time #Solving time 
        self.final_objective_function = self.objective_function(self.Q) #Final objective function
        
        self.Y = self.Y.reshape(-1,1) #Y array should have 2 dims as we need it to calculate the intercept
        support_indices = (self.alpha > self.eps) & (self.alpha < self.C).flatten() #Support vectors indices
        self.X_sv = self.X[support_indices] #X Support Vectors
        self.Y_sv = self.Y[support_indices] #Y Support Vectors
        self.alpha = self.alpha[support_indices]
        self.alpha = self.alpha.reshape(-1,1)
        

        self.intercept = np.mean(self.Y_sv - sum(self.alpha * self.Y_sv * self.kernel(self.X_sv,self.X_sv))) #Intercept
        self.alpha = self.alpha.reshape(-1,1) #Alpha array should have 2 dims as we need it to calculate for predict function

    #Predict function    
    def predict(self,x):
        return np.sign(np.sum(self.kernel(self.X_sv,x)*self.alpha*self.Y_sv,axis=0)+self.intercept)
    
