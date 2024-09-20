import numpy as np
import math
from scipy.integrate import quad
import statistics
from scipy import integrate
import random

######################            UCI data           ##################
data = np.genfromtxt('spectfheart.csv',
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')

x = data[:, :-1]
t = data[:, -1]
t[t == 0] = -1

import numpy as np
import pandas as pd
import time
import os

'''
                            nosie corrupted [OPTIONAL]
'''
import noiseADD as na
#x = na.withnoise(x, r=0.05)
#x = na.withnoise(x, r=0.1)
#x = na.withnoise(x, r=0.2)
x = na.withnoise(x, r=0.3)



from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)






from models import BFGSmoreauGPinLoss

'''
                            10-fold cross validation
'''
from sklearn.model_selection import KFold
n_folds = 5
cv = KFold(n_splits=n_folds, shuffle=True, random_state=58310798)

acc_train = np.empty((n_folds))
acc_test = np.empty((n_folds))
processingtime = np.empty((n_folds))
conf_M_train = np.empty((n_folds,2,2))
conf_M_test = np.empty((n_folds,2,2))
print('#'*75)
      
'''
                            Import evaluation package
'''
import time
from sklearn.metrics import accuracy_score, confusion_matrix


''' setting hyperparameter '''
m_    = 64
max_iter_ = 1000
tol_     = 0.001


''' SGDGPinLoss '''
tau_1_SGDGPinLoss = 0.5 
tau_2_SGDGPinLoss = 0.1 
epsl_1_SGDGPinLoss = 0.25
epsl_2_SGDGPinLoss = 0.5

''' RESmoreauGPinLoss '''
tau_1_RES = 0.7 
tau_2_RES = 0.4 
epsl_1_RES = 0.6
epsl_2_RES = 1
mu_RES   = 0.9
epsilon_0_ = 1e-2      #<---step size
tau_0_ = (10**6)     #<---step size
C_     = 1 
delta_ = 1 
gamma_ = 0.01 





'''
                            Testing with BFGSmoreauGPinLoss
'''
print("Using BFGSmoreauGPinLoss")
#start Cross Validation
for i, (train, test) in enumerate(cv.split(x)):
    #choose model     
    model = BFGSmoreauGPinLoss(C = C_, beta = 0.1, sigma = 0.1,
                mu = mu_RES, 
                tau_1 = tau_1_RES, 
                tau_2 = tau_2_RES, 
                epsl_1 = epsl_1_RES, 
                epsl_2 = epsl_2_RES,
                max_iter=max_iter_, tol=tol_)

    #steamp time
    start = time.time()
    #modeling
    model.fit(x[train], t[train])
    #cpu time used
    processingtime[i] = time.time() - start
    
    #prediction
    y_train = model.predict(x[train])
    y_test = model.predict(x[test])
     
    #evaluation model
    acc_train[i]= accuracy_score(t[train], y_train)
    acc_test[i] = accuracy_score(t[test], y_test)
    conf_M_train[i] = confusion_matrix(t[train], y_train)
    conf_M_test[i] = confusion_matrix(t[test], y_test)

#display result
print('CPU time = %.4f and its S.D. = %.4f' % (np.mean(processingtime), np.std(processingtime)))
print("Training accuracy = %.4f and its S.D. = %.4f" % (np.mean(acc_train), np.std(acc_train)))
print('Training confusion matirx: \n', np.mean(conf_M_train, axis=0))
print("Test accuracy     = %.4f and its S.D. = %.4f" % (np.mean(acc_test), np.std(acc_test)))
print('Test confusion matirx: \n', np.mean(conf_M_test, axis=0))
print('#'*75)






