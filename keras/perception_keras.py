import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam

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