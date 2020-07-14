# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:53:49 2020

@author: Salehin
"""


# For plotting
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns
from numpy import hstack
import numpy as np
from sklearn import metrics
sns.set_style("white")
#for matrix math
import numpy as np
#for normalization + probability density function computation
from scipy import stats
#for data preprocessing
import pandas as pd
from math import sqrt, log, exp, pi
from random import uniform
import csv
from sklearn import metrics

random_seed=36788765
np.random.seed(random_seed)

mu1 = 0  # Input parameter, mean of first normal probability distribution
sigma1 = 1.0 #@param {type:"number"}
mu2 = 0.1 # Input parameter, mean of second normal  probability distribution
sigma2 = 1.0 #@param {type:"number"}

true=[]
for i in range(1500):
    true.append(0)
for i in range(500):
    true.append(1)    

# generate data
y1 = np.random.normal(mu1, sigma1, 1500)
y2 = np.random.normal(mu2, sigma2, 500)
data=hstack((y1,y2))

# For data visiualisation calculate left and right of the graph
Min_graph = min(data)
Max_graph = max(data)
x = np.linspace(Min_graph, Max_graph, 2000) # to plot the data
#plt.figure()
#plt.subplot(2,2,1)
sns.distplot(data, bins=20, kde=False);
plt.show()


def pdf(datum,mu,sigma):
        "Probability of a data point given the current parameters"
        u = (datum - mu) / abs(sigma)
        y = (1 / (sqrt(2 * pi) * abs(sigma))) * exp(-u * u / 2)
        return y
    
def weight(data,mix):
    wp1=pdf(data,mu1,sigma1)*mix
    wp2=pdf(data,mu2,sigma2)*(1-mix)
    den=wp1+wp2
    wp1/=den
    wp2/=den
    return (wp1,wp2)

def loglikelihood(left,right,mix,mu1,sigma1,mu2,sigma2,data):
    s=0
    s1=0
    for i in range(len(left)):
       s+=left[i]*log(mix)+(right[i]*log(1-mix))
    for i in range(len(left)):
        s1+=left[i]*log(pdf(data[i],mu1,sigma1))+right[i]*log(pdf(data[i],mu2,sigma2))
    return s+s1    
mix=0.5
loglik=[]
tol=1e-5
j=0
loglik.append(0)
mixer=[]
while(True):
    left=[]
    right=[] 
    for i in range(len(data)):
        wt=weight(data[i],mix)
        left.append(wt[0])
        right.append(wt[1])    
    mix=sum(left)/len(data)
    mixer.append(mix)
    print(mix)
    d=loglikelihood(left.copy(),right.copy(),mix,mu1,sigma1,mu2,sigma2,data)
    loglik.append(d)
    if((j>1 and abs(loglik[j+1]-loglik[j]<tol))or j==1000):
        break
#    print(d)
    left.clear()
    right.clear() 
    j+=1
    
predict=[]
for i in range(len(left)):
    if (left[i]>right[i]):
        predict.append(0)
    if(left[i]<right[i]):
        predict.append(1)
correct=0        
for i in range(len(predict)):
    if (i>=0 and i<=len(left)-1):
        if(predict[i]==0):
             correct+=1
    else:
        if(predict[i]==1):
            correct+=1
print("SImiliarity Score: ",metrics.adjusted_rand_score(predict,true))            
#print('Correct Prediction Rate: ', correct/2000)                 
b=np.array(predict)   
np.savetxt('predict.txt',predict,delimiter=',',fmt='%1d')     
a=np.array(mixer)
np.savetxt('iter.txt',a,delimiter=',',fmt='%1.8f')  
loglik.pop(0)   
print('Number of iterations ', j+1)
#plt.subplot(2,2,2)
plt.plot(loglik) 
plt.xlabel('iteration')
plt.ylabel('log likelihood')  
plt.show() 
sample=data.reshape(len(data),1)
model=GaussianMixture(2,init_params='random')
model.fit(sample)
l=model.predict(sample)
dd=model.get_params()
print(model.lower_bound_)
data=data.reshape(-1,1)
s=model.score(data)

        
    
    
    