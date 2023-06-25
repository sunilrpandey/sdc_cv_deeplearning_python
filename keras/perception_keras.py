import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam



# plt.scatter(X[:n_pts,0], X[:n_pts,1]): This line creates a scatter plot using the scatter function from the matplotlib.pyplot module. It plots the data points from the first n_pts rows of the X array. X[:n_pts,0] selects the first n_pts rows and the first column of the X array, representing the x-coordinates of the data points. X[:n_pts,1] selects the first n_pts rows and the second column, representing the y-coordinates. This line is used to plot the data points of one class (e.g., class 0) in a binary classification problem.

# plt.scatter(X[n_pts:,0], X[n_pts:,1]): This line is similar to the previous line but plots the data points from the n_pts to the end of the X array. X[n_pts:,0] selects the rows from n_pts to the end and the first column of the X array, representing the x-coordinates of the data points. X[n_pts:,1] selects the rows from n_pts to the end and the second column, representing the y-coordinates. This line is used to plot the data points of the other class (e.g., class 1) in a binary classification problem.

# plt.show(): This line displays the scatter plot on the screen. It is necessary to include this line to show the plot in the notebook or console.

# model = Sequential(): This line creates a sequential model object using the Sequential class from Keras. The sequential model is a linear stack of layers.

# model.add(Dense(units=1, input_shape=(2,), activation='sigmoid')): This line adds a dense layer to the model using the Dense class from Keras. The units parameter is set to 1, indicating that the layer will have a single output neuron. The input_shape parameter is set to (2,), specifying that the layer expects inputs of shape (2,), corresponding to the two input features. The activation parameter is set to 'sigmoid', which applies the sigmoid activation function to the outputs of the layer.

# adam = Adam(lr = 0.1): This line creates an Adam optimizer object with a learning rate of 0.1. The Adam optimizer is commonly used for training neural networks.

# model.compile(adam, loss='binary_crossentropy', metrics=['accuracy']): This line compiles the model and configures the training process. It specifies the optimizer (adam), the loss function (binary_crossentropy), and the metrics to be evaluated during training (accuracy).

# h = model.fit(x=X, y=y, verbose=1, batch_size=50, epochs=500, shuffle='true'): This line trains the model using the fit function. It specifies the input data (X) and target labels (y) to be used for training. The verbose parameter is set to 1, which displays the training progress during each epoch. The batch_size parameter determines the number of samples per gradient update. The epochs parameter specifies the number of training iterations. The shuffle parameter is set to 'true', indicating that the training data should be shuffled before each epoch.

# This function plot_decision_boundary is used to plot the decision boundary of a binary classification model by creating a grid of points and calculating the predicted probabilities for each point. The contour plot shows the areas where the model predicts different classes, allowing for visualizing the decision boundary.

# x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1): This line creates a 1D array x_span using the linspace function from NumPy. It generates evenly spaced values between the minimum and maximum x-coordinates (X[:,0]) of the data points. By subtracting 1 from the minimum and adding 1 to the maximum, the range of x_span is slightly extended to ensure that the decision boundary is fully visible in the plot.

# y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1): This line is similar to the previous line but generates the y_span array for the y-coordinates (X[:,1]) of the data points.

# xx, yy = np.meshgrid(x_span, y_span): This line creates a grid of coordinates using the meshgrid function from NumPy. It takes two 1D arrays (x_span and y_span) and returns two 2D arrays (xx and yy). These arrays represent the x and y coordinates of all the points in the grid.

# xx_, yy_ = xx.ravel(), yy.ravel(): This line flattens the xx and yy arrays into 1D arrays using the ravel function. The flattened arrays (xx_ and yy_) represent all the points in the grid as separate coordinate pairs.

# grid = np.c_[xx_, yy_]: This line concatenates the xx_ and yy_ arrays column-wise using the np.c_ function from NumPy. The resulting grid array contains all the coordinate pairs from the grid.

# pred_func = model.predict(grid): This line uses the trained model to make predictions on the grid array. It calculates the predicted class probabilities for each point in the grid.

# z = pred_func.reshape(xx.shape): This line reshapes the pred_func array to match the shape of xx, resulting in a 2D array z with the same shape as the grid. The values in z represent the predicted class probabilities for each point in the grid.

# plt.contourf(xx, yy, z): This line creates a filled contour plot using the contourf function from matplotlib.pyplot. It visualizes the decision boundary by filling different regions of the plot with colors corresponding to the predicted class probabilities. The xx and yy arrays represent the coordinates of the grid, and z represents the predicted class probabilities for each point in the grid.

def trainModel(X,y,n_pts):
    # call to scatter for two sets of data, to get differnt colors
    plt.scatter(X[:n_pts,0], X[:n_pts,1])
    plt.scatter(X[n_pts:,0], X[n_pts:,1])
    plt.show() # If I dont show it points are blue/orange dots are coming and not syncing with Notebook
    
    model = Sequential()
    model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    adam = Adam(lr = 0.1)  # lr = learning rate
    model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(x=X, y=y, verbose=1, batch_size=50, epochs=500, shuffle='true')
    return h, model

def plotParam(h,param):
    plt.plot(h.history[param])
    plt.title(param)
    plt.xlabel('epoch')
    plt.legend([param])
    plt.show()

def plot_decision_boundary(n_pts, X, y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
    
    plt.scatter(X[:n_pts,0], X[:n_pts,1])
    plt.scatter(X[n_pts:,0], X[n_pts:,1])
    
    
def predictForValues(model, x, y):
    point = np.array([[x, y]])
    prediction = model.predict(point)
    plt.plot([x], [y], marker='o', markersize=10, color="red")
    plt.show()
    return prediction
 
def populateData():
    n_pts = 500
    np.random.seed(0)
    Xa = np.array([np.random.normal(13, 2, n_pts),
                np.random.normal(12, 2, n_pts)]).T
    Xb = np.array([np.random.normal(8, 2, n_pts),
                np.random.normal(6, 2, n_pts)]).T

    X = np.vstack((Xa, Xb))
    y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
    
    return n_pts, X, y
        
def demoPrediction():
    
    n_pts, X, y = populateData()
    
    h, model = trainModel(X,y,n_pts)    
    plotParam(h,'accuracy')
    plotParam(h,'loss')            
    plot_decision_boundary(n_pts, X, y, model)  
    
    prediction = predictForValues(model, x = 7.5, y = 5)    
    print("prediction is: ",prediction)

demoPrediction()