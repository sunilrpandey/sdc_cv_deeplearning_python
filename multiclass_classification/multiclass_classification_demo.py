import numpy as np
import keras
from sklearn import datasets
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def scatterData(class_size, X, y):
    for i in range(class_size):
        plt.scatter(X[y==i, 0], X[y==i, 1])
    

def populateData(n_pts, centers, class_size): 
    X, y = datasets.make_blobs(n_samples=n_pts, random_state = 123, centers=centers, cluster_std=0.4)
    
    scatterData(class_size, X, y)
    print(y)
    y_cat = to_categorical(y, 5)
    print(y_cat)
    
    return X , y, y_cat

def trainModel(X,y_cat ):
    model = Sequential()
    model.add(Dense(5, input_shape=(2,), activation='softmax'))
    model.compile(Adam(lr=0.1), 'categorical_crossentropy', metrics=['accuracy'])

    #one hot encode output
    history = model.fit(X, y_cat, verbose=1, batch_size = 50, epochs=100)
    return history, model

def plot_multiclass_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = np.c_[xx.ravel(), yy.ravel()]
    # pred_func = model.predict_classes(grid)
    predict_x=model.predict(grid) 
    pred_func=np.argmax(predict_x,axis=1)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
    
    
def predictAndShowData(model, x, y, clr):
    point = np.array([[x, y]])
    # prediction = model.predict_classes(point)
    predict_x = model.predict(point) 
    prediction = np.argmax(predict_x,axis=1)
    
    plt.plot([x], [y], marker = 'o', markersize = 10, color = clr)
    plt.show()
    return prediction
      
def demoMultiClassClassification():
    
    n_pts = 500
    centers = [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 0]]    
    class_size = len(centers)
    X , y, y_cat = populateData(n_pts, centers, class_size)
    history, model = trainModel(X,y_cat )
    plot_multiclass_decision_boundary(X, y, model)    
    
    scatterData(class_size,X,y)    
    prediction = predictAndShowData(model, 0.5, 1.5, "red")
    print("Prediction is: ", prediction)
    
demoMultiClassClassification()