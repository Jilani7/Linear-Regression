#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('pylab', 'inline')
plt.style.use('ggplot')
plt.rcParams['image.interpolation'] = 'nearest'


# In[2]:


X = np.arange(0,5,0.0001, dtype=np.float32)
delta = np.random.uniform(-1,1, size=X.shape[0])
Y = .4 * X + 3 + delta

#making a copy for later use
rx = X
ry = Y
print (len(X))


# In[3]:


plt.scatter(X,Y,s=0.01)
plt.xlabel('X')
plt.ylabel('Y')


# $$
# h(X, \theta) = X^T . \theta
# $$

# In[4]:


def hyp(theta, X):
    # Find the hypothesis when you receive 2 a matrix X and a vector theta, return the result as mentioned in the equation above.
    return theta.dot(X.T)


# $$
# cost = \frac{1}{2m} \sum_{i = 0}^m{(h(X^i, \theta)-Y^i)}^2
# $$

# In[5]:


def cost_function(theta,X,Y):
    # Find the cost function by using the above given equations 
    inner = np.power((X.dot(theta.T) - Y), 2)
    return np.sum(inner) / (2 * len(X))


# $$
# \frac{\delta}{\delta \theta_j} = \frac{1}{m} \sum_{i = 0}^m{(h(X^i, \theta)-Y^i)} * X_j
# $$

# In[6]:


def derivative_cost_function(theta,X,Y,alpha):
    m = len(Y)
    prediction = np.dot(X,theta)
    theta = theta -(1/m)*alpha*( X.T.dot((prediction - Y)))
    return theta


# In[7]:


print (X.shape)
nx=np.hstack((X,ones(len(X),)))
nx=nx.reshape((2,X.shape[0])).T
print (nx.shape)


# In[8]:


np.random.seed(20) # To make sure you have the same value as me
eps=0.0001
nexamples=float(nx.shape[0])
thetas=np.random.rand(nx.shape[1],)
print (thetas)


# In[9]:


cf=cost_function(thetas,nx,Y)
print (cf)


# In[10]:


ad=derivative_cost_function(thetas,nx,Y,0.01)
print (ad)


# In[11]:


def GradientDescent(X,Y,theta,maxniter=2000):
    """
    Write the gradient descent method in following order
    Find cost using the above function that you will build
    Find the new derivatives.
    If the cost is decreasing, continue the work, else break it
    """
    learning_rate = 0.01
    m = len(Y)
    cost_history = np.zeros(maxniter)
    theta_history = np.zeros((maxniter,2))
    for i in range(maxniter):
        prediction = np.dot(X,theta)
        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - Y)))
        theta_history[i,:] =theta.T
        cost_history[i]  = cost_function(theta,X,Y)
        print ("Iteration: " + str(i) + " with theeta: " + str(theta) + " has cost : " + str(cost_history[i]))
    return theta


# In[12]:


# NO need to pass other arguments we can directly call them in GradientDecent
theta_new = GradientDescent(nx,Y,thetas)
print(theta_new)


# In[13]:


plt.scatter(X,Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(nx[:,0],np.dot(nx,theta_new),c='0.1')


# In[ ]:





# In[ ]:





# In[ ]:




