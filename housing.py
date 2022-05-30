import numpy
import matplotlib.pyplot as plt 

#loading the data
train_data = numpy.loadtxt("dataset/boston_train.csv", delimiter= ",")
test_data = numpy.loadtxt("dataset/boston_test.csv", delimiter= ",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

#reshaping numpy arrays to ensure we have N-dimensional Numpy arrays 
t_train = numpy.reshape(len(t_train), 1) 
t_test = numpy.reshape(len(t_test), 1)

#number of instances
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features %i" % X_train.shape[1])


