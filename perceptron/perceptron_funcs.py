import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

 
def showData():
    n_pts=100
    np.random.seed(0)
    # This line generates random data points for the top region. np.random.normal(10,2,n_pts) generates n_pts 
    # random numbers from a normal distribution with mean 10 and standard deviation 2. 
    # Similarly, np.random.normal(12,2,n_pts) generates n_pts random numbers from a normal distribution with 
    # mean 12 and standard deviation 2. These two arrays are passed to np.array to create a 2D array with shape (n_pts, 2). 
    # The .T at the end transposes the array, switching the rows and columns. The resulting array top_region represents the coordinates of the data points in the top region.
    top_region=np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts)]).T
    bottom_region= np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts)]).T
    _, ax= plt.subplots(figsize=(4,4))
    # This line creates a scatter plot of the data points in the bottom region. bottom_region[:,0] 
    # selects the first column of the bottom_region array, representing the x-coordinates of the data points. 
    # bottom_region[:,1] selects the second column, representing the y-coordinates. 
    ax.scatter(top_region[:,0], top_region[:,1], color='r')
    ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
    
    plt.show()

    
def draw(x1,x2):
  ln=plt.plot(x1,x2)

def draw_new(x1,x2):
  ln=plt.plot(x1,x2,'-')
  plt.pause(0.0001)
  ln[0].remove()
 
def sigmoid(score):
  return 1/(1+np.exp(-score))

def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid(points * line_parameters)
    # print(np.log(p).shape)
    cross_entropy = -(1 / m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy

def gradiant_discent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(500):
        p = sigmoid(points * line_parameters)
        gradiant_discent = (points.T * (p - y)) * (alpha / m)
        line_parameters = line_parameters - gradiant_discent
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        
        x1=np.array([points[:,0].min(), points[:,0].max()])
        x2= -b/w2 + (x1*(-w1/w2)) 
        draw_new(x1,x2)
        print(calculate_error(line_parameters,points,y))
        
        
def demoErrorFuncGradiantDiscent():
    n_pts=100
    np.random.seed(0)
    bias= np.ones(n_pts)
    top_region=np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts), bias]).T
    bottom_region= np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts), bias]).T
    all_points=np.vstack((top_region, bottom_region))
    line_paramters = np.matrix([np.zeros(3)]).T
    y = np.array([np.zeros(n_pts),np.ones(n_pts)]).reshape(n_pts * 2,1)
    
    _, ax= plt.subplots(figsize=(4,4))
    ax.scatter(top_region[:,0], top_region[:,1], color='r')
    ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
    gradiant_discent(line_paramters,all_points,y,0.06) 
    plt.show()

def demoErrorFunc():
    n_pts=10
    np.random.seed(0)
    bias= np.ones(n_pts)
    top_region=np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts), bias]).T
    bottom_region= np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts), bias]).T
    all_points=np.vstack((top_region, bottom_region))
    #print("line_paramters.shape -> ", all_points)
    w1 = -0.1    #-0.2
    w2 = -0.15   #-0.35
    b = 0       #  3.5
    line_paramters = np.matrix([w1,w2,b]).T
    #print("line_paramters.shape -> ", line_paramters.shape)
    
    x1=np.array([bottom_region[:,0].min(), top_region[:,0].max()])
    #print("x1 -> ", x1)
    x2= -b/w2 + (x1*(-w1/w2)) 
    
    
    linear_combination= all_points*line_paramters 
    probabilities= sigmoid(linear_combination)
    #print(probabilities)    
    #print("all_points->\n", all_points)
    y = np.array([np.zeros(n_pts),np.ones(n_pts)]).reshape(n_pts *2,1)
    # print(y.shape)
    print(calculate_error(line_paramters,all_points,y))
    
    _, ax= plt.subplots(figsize=(4,4))
    ax.scatter(top_region[:,0], top_region[:,1], color='r')
    ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
    draw(x1,x2)
    plt.show()

demoErrorFuncGradiantDiscent()
# demoErrorFunc()