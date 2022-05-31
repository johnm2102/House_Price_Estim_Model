import numpy as np
import matplotlib.pyplot as plt 

#loading the data
train_data = np.loadtxt("dataset/boston_train.csv", delimiter= ",")
test_data = np.loadtxt("dataset/boston_test.csv", delimiter= ",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
mean = t_train.mean()
print("Mean price: %f" % mean)

# (b) RMSE function
def rmse(t, tp):
    t = t.reshape((len(t), 1))
    tp = tp.reshape((len(tp), 1))
    return np.sqrt(np.mean((t - tp)**2))
# mean predictions
preds = mean * np.ones(len(t_test))
err = rmse(preds, t_test)
print("RMSE using mean predictor: %f" % err)

# (c) visualization of results
plt.scatter(t_test, preds)
plt.xlabel("House Prices")
plt.ylabel("Predicted House Prices")
plt.xlim([0,50])
plt.ylim([0,50])
plt.title("Mean Estimator (RMSE=%f)" % err)
plt.show()




    
