import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    Uni = (np.unique(y))    
    Mean = []
    
    #References:
    # Covariance https://en.wikipedia.org/wiki/Covariance_matrix
    #https://www.youtube.com/watch?v=9IDXYHhAfGA
    
    #
    
    covmat = np.cov(X.T) ## .T is used to create the transpose
    for i in Uni:
        x = []
        for j in range(len(y)):
            if(i == y[j]):
                x.append(X[j])
        Mean.append(np.mean(x,axis= 0))
    means = np.array(Mean).T
    #print(means)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    Uni = (np.unique(ytest))
    Means = []
    covmats = []

    #Find mean of all Xs for unique y values and covariance wrt those Xs

    for i in Uni:
        x = []
        for j in range(len(y)):
            if(i == y[j]):
                x.append(X[j])
        Means.append(np.mean(x,axis= 0))
        covmats.append(np.cov(np.array(x).T))
    means = np.array(Means).T
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # reference Lecture 4 page 15
    yEstimate = [] 
    Uni = np.unique(y)
    for x in Xtest:
        Max_prob = []
        for i in  range(len(Uni)):
            u = means[:,i].reshape(1,2)
            z =  ( 1/(((2*np.pi)**(x.shape[0]/2))*(det(covmat)**0.5)))*np.exp(-1 / 2 * np.dot((x-u), np.dot(inv(covmat), (x-u).T))) #determinant
            Max_prob.append([z])
            
        for i in range(len(Max_prob)):
            if Max_prob[i] == max(Max_prob):
                yEstimate.append(Uni[i])
    
    # Calculate accuracy in terms of matches.
    count = 0
    for i in range(len(yEstimate)):
        if(yEstimate[i]== ytest[i]):
            count +=1

    acc = count/len(yEstimate)
    
    ypred = np.array(yEstimate)
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # reference Lecture 4 page 15
    yEstimate = []
    Uni = np.unique(y)
    


    for x in Xtest:
        label_prob = []
        #print(x)
        for i in range(len(Uni)):
            u = means[:,i]
            #print(u.shape)
            z = ( 1/(((2*np.pi)**(x.shape[0]/2))*(det(covmats[i])**0.5))) * (np.exp(-1 / 2 * np.dot((x-u), np.dot(inv(covmats[i]), (x-u).T))))
            label_prob.append([z])
            
        for i in range(len(label_prob)):
            if label_prob[i] == max(label_prob):
                yEstimate.append(Uni[i])
    #print(means,means[:,1],label_list)
    #Accuracy
    accurate = 0
    for i in range(len(yEstimate)):
        if(yEstimate[i]== ytest[i]):
            accurate +=1
    
    acc = accurate / len(yEstimate)
    ypred = np.array(yEstimate)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    # Reference Lecture 5 page 8
    XtX = np.dot(X.T,X)
    Xty = np.dot(X.T,y)
    w = np.dot(inv(XtX),Xty)
                                                  
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    # Referred Lecture notes 5 page 19
    a = np.dot(X.T, X)
    d = a.shape[0]
    b = lambd*np.identity(d)
    c = inv(a+b)
    w = np.dot(c, np.dot(X.T,y))
                                                  
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    # Reference
    # https://medium.com/data-science-bootcamp/understand-dot-products-matrix-multiplications-usage-in-deep-learning-in-minutes-beginner-95edf2e66155
    yEst = np.dot(Xtest,w)
    error = ytest - yEst
    mse = np.dot(error.T,error)/(len(error))
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    # # Referred Lecture notes 5 page 19
    
    w = w.reshape(X.shape[1],1)
    error = ((np.dot((y - np.dot(X,w)).T,(y - np.dot(X,w))) + lambd*np.dot(w.transpose(),w)))/2
    
    # differentiating error wrt to w
    # -(X.Ty)+(X.TXw)+lambda*w
    z = np.dot(X,w)
    a = np.dot(X.T,z)
    b = -(np.dot(X.T,y))
    error_grad = b + a + lambd*w
    error_grad = np.squeeze(np.array(error_grad))
    #print(error_grad.shape)                                             
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1))
    N = x.shape[0]
    Xp = np.zeros((N,(p+1)))
    #print(Xp.shape)
    #Xp = np.ones((N,(p+1))) 
    for i  in range(p+1):
        # Here x**0 = 1, x**1 = x, x**2 = x^2 untill x**p = x^p
        Xp[:,i] = (x**i)
	
    # IMPLEMENT THIS METHOD
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()
#print(min(mses4))


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
