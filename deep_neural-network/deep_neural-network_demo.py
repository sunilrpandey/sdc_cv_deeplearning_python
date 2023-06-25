import numpy as np
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# model.add(Dense(4, input_shape=(2,), activation='sigmoid')): 

# - The first argument specifies the number of neurons (or units) in the layer. In this case, the dense layer has 4 neurons.
# input_shape=(2,): The input_shape argument defines the shape of the input data. Since this is the first layer of the model, we provide the input shape here. In this case, the input shape is (2,), which means the model expects inputs of size 2 (two features).
# activation='sigmoid': The activation argument specifies the activation function to be used in the layer. The 'sigmoid' activation function is used, which squashes the output values between 0 and 1, allowing the layer to model non-linear relationships.
# model.add(Dense(1, activation='sigmoid')): This line adds another dense layer to the model object. Here's the breakdown:

# - The number of neurons in this layer is set to 1. Since this is the last layer in the model, it represents the output layer. In this case, we have a binary classification problem, so the output layer has a single neuron representing the probability of the positive class.
# activation='sigmoid': Again, the 'sigmoid' activation function is used for the output layer to produce probabilities between 0 and 1.

def plot_decision_boundary(n_pts, X, y, model):
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25)
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

def plotParam(h,param):
    plt.plot(h.history[param])
    plt.title(param)
    plt.xlabel('epoch')
    plt.legend([param])
    plt.show()

def trainModel(X,y,n_pts):
    model = Sequential()
    model.add(Dense(4, input_shape=(2,), activation='sigmoid')) # Hidden layer 1, 4 perceptrons/neuron, 2 input
    model.add(Dense(1, activation='sigmoid'))# Output layer, 1 perceptrons, 2 input
    model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics=['accuracy'])

    h = model.fit(x=X, y=y, verbose=1, batch_size=20, epochs=100, shuffle='true')
    return h, model

    
def populateData():
    n_pts = 500
    X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    
    return n_pts, X, y

def demoDeepNeuralNetwork():
    n_pts, X, y = populateData()
    plt.show()
    
    h, model = trainModel(X,y,n_pts)    
    
    plotParam(h,'accuracy')
    plotParam(h,'loss')            
    
    plot_decision_boundary(n_pts, X, y, model)  
    prediction = predictForValues(model, x = 0.5, y = 0.6)    
    print("prediction is: ",prediction)
    
    plot_decision_boundary(n_pts, X, y, model)  
    prediction = predictForValues(model, x = 0, y = 0)    
    print("prediction is: ",prediction)

demoDeepNeuralNetwork()

    
    
    
    
    
