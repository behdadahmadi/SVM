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



def CrossValidation(all_folds,C,lambda_param):
    '''
        #####################################################
                    Cross validation using k-folds
        #####################################################
        folds,C and lambda parameter to cross validate the model
        it returns validation_mean_accuracy and training_mean_accuracy
    '''
    n_folds = all_folds.shape[0] #Number of folds
    validation_accuracy = []#Accruacy of model performed by each folds
    training_accuracy = [] #Accuracy of training
    for f in range(n_folds): #For each fold
        folds = np.arange(n_folds) #An array of number of folds
        training_folds = np.delete(folds,f,axis=0) #Removing the validation fold and create a new array
        training_folds = np.vstack(all_folds[training_folds]) #Training fold
        validation_fold = all_folds[f] #Validaiton array
        validationX,validationY = validation_fold[:,:-1],validation_fold[:,-1] 
        trainingX,trainingY = training_folds[:,:-1],training_folds[:,-1].reshape(-1,1)
        model = SVM(C=C,lambda_param=lambda_param)
        model.fit(trainingX,trainingY)
        predictions = model.predict(validationX)
        valid_acc = Accuracy(validationY,predictions)
        training_acc = Accuracy(trainingY,model.predict(trainingX))
        validation_accuracy.append(valid_acc)
        training_accuracy.append(training_acc)
    validation_mean_accuracy = np.mean(np.array(validation_accuracy))
    training_mean_accuracy = np.mean(np.array(training_accuracy))
    return C,lambda_param,validation_mean_accuracy,training_mean_accuracy
    
def GridSearch(folds):
    '''
        #####################################################
              Performing Grid search to find parameters
        #####################################################
        folds as input, and it run grid search with predifined C and lambda parameters ranges
        in total 2100 combination to evaluate
    '''
    C = np.arange(0.1,10.1,0.1) #An array from 0.1 to 10, step size 0f 0.1 for C parameter
    lambda_params = np.arange(1,3.1,0.1) #An array from 1 to 3, step size of 0.1 for lambda parameter
    grid = np.array(np.meshgrid(C,lambda_params)).T.reshape(-1,2) #2100 combination
    result = [] #To store the cross validation result
    for g in grid:
        #g[0] is C , g[1] is lambda parameter
        c,lp,val_acc,train_acc = CrossValidation(folds,g[0],g[1])
        result.append([c,lp,val_acc,train_acc])
        print(c,lp,val_acc,train_acc)
    return result


class SVM:
    '''
        #####################################################
            Soft margin SVM, solving with CVXOPT library
        #####################################################
        Parameters to input: C and Lambda
        Kernel: Polynomial
    '''
    
    def __init__(self,C,lambda_param,kernel="poly"):
        self.C = C #C parameter
        self.lambda_param = lambda_param #Lambda param of polynomial kernel
        self.alpha = None #Alphas
        self.intercept = 0 #Intercept inited with 0
        self.X = None #Features
        self.Y = None #Labels
        self.X_sv = None #X support vectors
        self.Y_sv = None #Y support vectors
        self.samples = 0 #No. of samples
        self.features = 784 #Number of features which is 28*28 = 784 pixels
        self.solve_time = 0 #Optimization time
        self.weights = None #Coefficients
        self.n_iterations = None #Number of solver iterations
        self.n_sv = None #Number of support vectors
        self.final_objective_function = None
        if kernel == "poly":
            self.kernel = lambda x1,x2: np.power(np.dot(x1, x2.T) + 1, self.lambda_param)
        elif kernel == "rbf":
            self.kernel = lambda x1,x2: self.gaussian_kernel(x1,x2)

    
    def gaussian_kernel(self,x1,x2):
        a1 = np.sum(x1**2,axis=1).reshape(-1,1)
        a2 = np.sum(x2**2,axis=1)
        return np.exp(-1*self.lambda_param * (a1+a2 - 2*(x1.dot(x2.T))))
    

  
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
        #I should declare that, at beginning I chose eps = 1e-3, but with lambda=3 for the kernel,
        # I got NaN values, therefore I changed it to 1e-6 to solve it
        
        #Setting solver tolerance options to getting better result
        cvx.solvers.options['abstol'] = 1e-13 #Absolute tolerance
        cvx.solvers.options['reltol'] = 1e-13 #Relative tolerace
        cvx.solvers.options['feastol'] = 1e-13 #Feasibility tolerance
        cvx.solvers.options['show_progress'] = False #Setting show_progress option to False to avoid showing progress of solver

        #Creating necessary paramters to pass to the solver
        K = self.kernel(self.X, self.X) #Gram matrix
        P = cvx.matrix(np.matmul(self.Y,self.Y.T) * K,tc="d")
        q = cvx.matrix(-np.ones((self.samples, 1)),tc="d")
        G = cvx.matrix(np.vstack((np.eye(self.samples)*-1,np.eye(self.samples))),tc="d")
        h = cvx.matrix(np.hstack((np.zeros(self.samples), np.ones(self.samples) * self.C)),tc="d")
        A = cvx.matrix(self.Y.reshape(1, -1),tc="d")
        b = cvx.matrix(np.zeros(1),tc="d")

        '''
                #####################################################
                                Running the Solver
                #####################################################
                Running the solver, getting alphas and support vectors, calculating intercept
        '''
        t1 = time() #Starting point
        solver_result =cvx.solvers.qp(P,q,G,h,A,b) #Solver result
        self.solve_time = time() - t1
        self.final_objective_function = solver_result["dual objective"]
        self.n_iterations = solver_result["iterations"] #Number of iterations
        alphas = np.array(solver_result["x"]) #Alphas without being checked by Lagrange multipliers
        #Checking for Lagrange multipliers
        support_indices = (alphas > eps).flatten() #Support vectors (these are indices of the vectors)
        self.alpha = alphas[support_indices] #Alphas
        self.X_sv = self.X[support_indices] #X Support Vectors
        self.Y_sv = self.Y[support_indices] #Y Support Vectors
        self.n_sv = len(self.X_sv)

        #Intercept
        self.intercept = np.mean(self.Y_sv - sum(self.alpha * self.Y_sv * self.kernel(self.X_sv,self.X_sv))) #Intercept

        '''
                #####################################################
                                KKT Conditions checking
                #####################################################
                Computing small m and big M
        '''
        Y = Y.reshape(-1,1) #Reshaping label data
        alphas = alphas.ravel() #Flatting alphas (retrieved directly from solver result)
        y_k = np.outer(Y,Y) * K #Outter product of Y data muliply by K matrix 
        Y = Y.ravel() #Flattening the Y array
        gradient_y= -1 * (y_k.dot(alphas) - 1)/Y #Negative gradient divided by Y(label data)

        #Calculating R and S sets
        R = np.where((alphas < eps) & (Y == +1) | (alphas > self.C-eps) & (Y == -1) | (alphas > eps) & (alphas < self.C-eps))[0]
        S = np.where((alphas < eps) & (Y == -1) | (alphas > self.C-eps) & (Y == +1) | (alphas > eps) & (alphas < self.C-eps))[0]
        self.m = max(gradient_y[R]) #Small m
        self.M = min(gradient_y[S]) #Big M
        self.kkt_condition = self.m - self.M #KKT Condition equation
    #Predict funtion
    def predict(self,x):
        return np.sign(np.sum(self.kernel(self.X_sv,x) * self.alpha * self.Y_sv,axis=0) + self.intercept)


