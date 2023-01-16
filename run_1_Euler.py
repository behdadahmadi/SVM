from functions_1_Euler import *

#-----Loading,prepare the data
xLabel1,xLabel5 = get_data() #Getting and loading data
X,Y = PreProcessData([xLabel1,xLabel5]) #Preprocessing the data and creating labels array
trainX,testX,trainY,testY = CreateTrainTestSplit(X,Y) #Splitting whole data to  trainset and testset by 80% and seed = 1985602
folds = KFoldsSplit(trainX,trainY,kfolds=5) #Creating 5 folds to perform GridSearch
#-----
#-----Grid searching
#Kindly, please uncomment this line to perform GridSearch
#grid_search = GridSearch(folds)

#-----initlizing the model and fiting it
model = SVM(0.1,2.4,kernel="poly") #C = 0.1 , lambda_parameter = 2.4
model.fit(trainX,trainY) #Fitting the model
#-----
#-----Getting the information to print
training_prediction = model.predict(trainX) #Prediction values of training dataset
testing_predictions = model.predict(testX) #Prediction values of testing dataset
training_accuracy = Accuracy(training_prediction,trainY) #Training accuracy
testing_accuracy = Accuracy(testing_predictions,testY) #Testing accuracy
cfm = ConfusionMatrix(testing_predictions,testY) #Confusion Matrix
C_param = 0.1; lambda_param = 2.4
solving_time = model.solve_time #Solving time in seconds
number_of_iterations = model.n_iterations #Number of iterations
m_M_diff = model.m - model.M

#-----
#-----Printing
print("Polynomial kernel is used and hyper parameter values are:  C={} , Lambda parameter= {}".format(C_param,lambda_param))
print("Classification rate on training set: {}%".format(training_accuracy*100.0))
print("Classification rate on test set: {}%".format(testing_accuracy*100.0))
print("The confusion matrix:")
print(cfm)
print("Optimization time: {}".format(solving_time))
print("Number of iterations of solver: {}".format(number_of_iterations))
print("The different between m(a) and M(A) = {}".format(m_M_diff))