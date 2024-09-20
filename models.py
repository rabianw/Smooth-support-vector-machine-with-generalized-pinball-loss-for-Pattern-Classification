import numpy as np
import math
from scipy.integrate import quad
from scipy import integrate
import random
from sklearn.base import BaseEstimator
import numexpr as ne




class BFGSmoreauGPinLoss(BaseEstimator):
    #setting initial parameters    
    def __init__(self, C = 1, beta = 0.1, sigma = 0.1, alpha_0 = 1,
                 mu = 0.99, tau_1 = 1, tau_2 = 1, 
                 epsl_1 = 0.25, epsl_2 = 0.5,
                 max_iter=1000, tol=0.001):
        
        assert C > 0
        self.C = C
        assert 0 < beta  < 1
        self.beta = beta
        assert 0 < sigma  < 1
        self.sigma  = sigma 
        assert 0 <= mu < 1
        self.mu = mu
        assert tau_1 > 0
        self.tau_1 = tau_1
        assert tau_2 > 0
        self.tau_2 = tau_2
        assert epsl_1 > 0
        self.epsl_1 = epsl_1
        assert epsl_2 > 0
        self.epsl_2 = epsl_2
        assert max_iter > 0 and isinstance(max_iter, int)
        self.max_iter=max_iter
        assert tol > 0 
        self.tol=tol
        self.alpha_0 = alpha_0


    ''' cost function (squareHingeloss SVM) '''
    def cost_function(self,w,x,t):
        u = (1-t*np.matmul(x,w))
        p = np.piecewise(u, [
                         u >= (self.epsl_1/self.tau_1) + self.tau_1*self.mu,
                        ((self.epsl_1/self.tau_1) <= u) & (u <= (self.epsl_1/self.tau_1) 
                        + self.tau_1*self.mu),
                        (-(self.epsl_2/self.tau_2) <= u) & (u <= (self.epsl_1/self.tau_1)),
                        (-(self.epsl_2/self.tau_2) - self.tau_2*self.mu <= u) & 
                        (u <= -(self.epsl_2/self.tau_2))
                    ], [
                        lambda u: self.tau_1*(u - (self.epsl_1/self.tau_1))- (((self.tau_1)**2)*self.mu)/2 ,
                        lambda u: (u - (self.epsl_1/self.tau_1))**2/(2*self.mu),
                        0,
                        lambda u: ((u + (self.epsl_2/self.tau_2))**2)/(2 * self.mu),
                        lambda u:   -self.tau_2*(u + (self.epsl_2/self.tau_2))-(((self.tau_2)**2)*self.mu)/2
                    ])
        
        loss = self.C*(np.mean(p))
        cost = 1/2 * np.linalg.norm(w)**2  + loss
        return cost

    ''' approximate ResNaq_BFGS matrix '''
    def Update_RES_BFGS(self,B, dw, dg, I):
            dg_t =  dg[:, np.newaxis]
               
            Bdw = np.dot(B, dw)
            Bdw = np.dot(B, dw)
            dw_t_B = np.dot(dw, B)
            dwBdw = np.dot(np.dot(dw, B), dw)

            p = dg_t*dg
            u = Bdw[:, np.newaxis] * dw_t_B

            B_new = B + p / np.dot(dg, dw) - u / dwBdw 
            return p, u, B_new

    def update_step_size(self,w,d,grad,x,t) -> np.ndarray:
        """ Armijo rule. """

        h = lambda a : self.cost_function(w + a * d,x,t) - \
                       self.cost_function(w,x,t)
        assert h(0) == 0.

        # TODO: What if g >= 0 ? set g = 0 ?
        g = grad.dot(d)

        alpha = self.alpha_0

        if h(alpha) > self.sigma * g * alpha:
            # Decrease alpha.
            while True:
                alpha = alpha * self.beta
                if h(alpha) <= self.sigma * g * alpha:
                    self.alpha = alpha
                    break
        else:
            # Increase alpha.
            while True:
                prev_alpha = alpha
                alpha = alpha / self.beta
                if h(alpha) > self.sigma * g * alpha:
                    self.alpha = prev_alpha
                    break

        return self.alpha
        
        
    def fit(self, x, t):

        # checks for labels
        self.classes_ = np.unique(t)
        t[t==0] = -1

        #initail variables k, stop, w_0
        k = 1
        update = True
        w = np.zeros(len(x[0])+1)
        w_ = np.zeros(len(x[0])+1)
        
        
        #set up matrices
        B = np.identity(len(x[0])+1)  #inverse hessian
        I = np.identity(len(x[0])+1)  #identity 
        X = np.ones((len(x),len(x[0])+1))
        X[:,:-1] = x 

        obj_res = []
        iter_res = []

        
        
        while k <=(self.max_iter) and update :

                # compute cost function
                cost = self.cost_function(w,X,t)
                obj_res.append(cost)
             

                X_ = X
                t_ = t

                #compute gradient
                u = (1-t_*np.matmul(X_,w))
                p = np.piecewise(u, [
                    u >= (self.epsl_1/self.tau_1) + self.tau_1*self.mu,
                    ((self.epsl_1/self.tau_1) <= u) & (u <= (self.epsl_1/self.tau_1) 
                    + self.tau_1*self.mu),
                    (-(self.epsl_2/self.tau_2) <= u) & (u <= (self.epsl_1/self.tau_1)),
                    (-(self.epsl_2/self.tau_2) - self.tau_2*self.mu <= u) & 
                    (u <= -(self.epsl_2/self.tau_2))
                ], [
                    self.tau_1,
                    lambda u: (u - (self.epsl_1/self.tau_1))/(self.mu),
                    0,
                    lambda u: (u + (self.epsl_2/self.tau_2))/(self.mu),
                    -self.tau_2
                ])
                #print('grad of loss is', p)
                grad = w - self.C*(np.mean(p*t_*X_.T,axis=1))

                
                if  np.linalg.norm(grad)<=self.tol: 
                    update = False
                    print('DONE')
                else:
                    
                    # update hessian matrix
                    H = np.linalg.inv(B)

                    #update step size (Amijo)
                    d = -np.matmul(H,grad)                          
                    eta = self.update_step_size(w,d,grad,X_,t_)

                    print( 'eta=',eta)
                    
                    # update weight
                    w_new = w - eta * np.matmul(H,grad) 

                    #compute new gradient
                    u = (1-t_*np.matmul(X_,w_new))
                    p = np.piecewise(u, [
                        u >= (self.epsl_1/self.tau_1) + self.tau_1*self.mu,
                        ((self.epsl_1/self.tau_1) <= u) & (u <= (self.epsl_1/self.tau_1) 
                        + self.tau_1*self.mu),
                        (-(self.epsl_2/self.tau_2) <= u) & (u <= (self.epsl_1/self.tau_1)),
                        (-(self.epsl_2/self.tau_2) - self.tau_2*self.mu <= u) & 
                        (u <= -(self.epsl_2/self.tau_2))
                    ], [
                        self.tau_1,
                        lambda u: (u - (self.epsl_1/self.tau_1))/(self.mu),
                        0,
                        lambda u: (u + (self.epsl_2/self.tau_2))/(self.mu),
                        -self.tau_2
                    ])
                    #print('grad of loss is', p)
                    grad_new = w_new - self.C*(np.mean(p*t_*X_.T,axis=1))

                    # get difference of values
                    dw = w_new - w 
                    # get difference of gradients
                    dg = grad_new - grad 

                    _, _, B_new = self.Update_RES_BFGS(B, dw, dg, I)

                    # update step
                    B = B_new
                    w = w_new
                   


                    k += 1
                    iter_res.append(k) 
                  
               

                    
        self.final_iter = k
        self._coef = w[:-1]
        self._intercept = w[-1]
        self.obj_res = obj_res
        self.iter_res = iter_res
        
    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)






class SGDHingeLoss(BaseEstimator):

    #setting initial parameters
    def __init__(self, C=0.5,m =10, max_iter=10000, tol=0.0001):
        self.C = C
       # self.eta=eta
        self.max_iter=max_iter
        self.tol=tol
        self.m = m

    ''' cost function (squareHingeloss SVM) '''
    def cost_function(self,w,x,t):

        u = 1-t*np.matmul(x,w)
        p = np.sign(u)
        p[p==-1] = 0
               
        loss = self.C*(np.mean(p))
        cost = 1/2 * np.linalg.norm(w)**2  + loss
        return cost
        
    def fit(self, x, t):
        #initail variables k, stop, w_0
        k = 1
        update = True
        w = np.zeros(len(x[0])+1)
        w_ = w
        #print('w_0 is ', w)
        #set up matrices
        X = np.ones((len(x),len(x[0])+1))
        X[:,:-1] = x 

        obj_sgdh = []
        iter_sgdh = []
        
        while k <=(self.max_iter) and update :

            # compute cost function
            cost = self.cost_function(w,X,t)
            obj_sgdh.append(cost)

            #stochastic random a datum
            r = np.random.randint(len(X),size=self.m) 
            #compute gradient
            u = 1-t[r]*np.matmul(X[r,:],w)
            p = np.sign(u)
            p[p==-1] = 0
            grad = w - self.C*np.mean(p*t[r]*X[r,:].T,axis=1)  
            #print('norm grad is', np.linalg.norm(grad))
            #check stopping criterion
            if np.linalg.norm(grad)<=self.tol: 
                self.final_iter = k
                update = False
                print('DONE')
                
            else:
                #update step
                self.eta = 1/(k)
                w -= self.eta*grad
                k += 1

                #collect output from each iteration
                w_ += w/k

                iter_sgdh.append(k) 

                #print('(k,w) is', (k,w_))
                
        #keep the final solutions
        self.final_iter = k
        self._coef = w_[:-1]
        self._intercept = w_[-1]
        self.obj_sgdh = obj_sgdh
        self.iter_sgdh = iter_sgdh
        
    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p


class SGDGPinLoss(BaseEstimator):
    #setting initial parameters
    def __init__(self, C=0.5 ,m=50 , tau_1=0.65, epsl_1=1, tau_2=0.75, epsl_2=0.5, max_iter=5, tol=0.001):
        self.C = C
        self.m = m
        self.tau_1=tau_1
        self.epsl_1=epsl_1
        self.tau_2=tau_2
        self.epsl_2=epsl_2
      
        self.max_iter=max_iter
        self.tol=tol

    ''' cost function (squareHingeloss SVM) '''
    def cost_function(self,w,x,t):


        u = 1-t*np.matmul(x,w)
        p = np.piecewise(u, 
                        [u>self.epsl_1/self.tau_1,
                        (-self.epsl_2/self.tau_2 <= u) & (u<=self.epsl_1/self.tau_1),
                        u<-self.epsl_2/self.tau_2], 
                        [
                        lambda u: self.tau_1 * (u-self.epsl_1/self.tau_1),
                        0,
                        lambda u: -self.tau_2 * (u-self.epsl_2/self.tau_2)])
               
       


        loss = self.C*(np.mean(p))
        cost = 1/2 * np.linalg.norm(w)**2  + loss
        return cost

    
        
    def fit(self, x, t):
        #initail variables k, stop, w_0
        k = 1
        update = True
        w = np.zeros(len(x[0])+1)
        w_ = w
        #print('w_0 is ', w)
        #set up matrices
        X = np.ones((len(x),len(x[0])+1))
        X[:,:-1] = x 

        obj_sgdG = []
        iter_sgdG = []
        
        while k <=(self.max_iter) and update :
            # compute cost function
            cost = self.cost_function(w,X,t)
            obj_sgdG.append(cost)

            #stochastic random a datum
            r = np.random.randint(len(X),size=self.m)                       

            #compute gradient
            u = 1-t[r]*np.matmul(X[r,:],w)
            p = np.piecewise(u, 
                             [u>self.epsl_1/self.tau_1,
                              (-self.epsl_2/self.tau_2 <= u) & (u<=self.epsl_1/self.tau_1),
                              u<-self.epsl_2/self.tau_2], 
                             [self.tau_1,
                              0,
                              -self.tau_2])
            
            grad = w - self.C*np.mean(p*t[r]*X[r,:].T,axis=1) 
            #มอง p เป็น row เมทริก เพื่อมาคูณแบบ pointwise กับ X
            #ตอนหาค่าเฉลี่ย หาค่าในแนว row
            #print('norm grad is', np.linalg.norm(grad))
            #check stopping criterion
            if np.linalg.norm(grad)<=self.tol: 
                self.final_iter = k
                update = False
                print('DONE')
                
            else:
                #update step
                self.eta = 1/(k)
                w -= self.eta*grad
                k += 1
                iter_sgdG.append(k) 
                #projection step (OPTIONAL)
               # w  = min(1,1/np.sqrt(self.l)/np.linalg.norm(w))*w
                #collect output from each iteration
                w_ += w/k

                #print('(k,w) is', (k,w_))
                
        
        self.final_iter = k
        self._coef = w_[:-1]
        self._intercept = w_[-1]
        self.obj_sgdG = obj_sgdG
        self.iter_sgdG = iter_sgdG
        

    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)


class SGDmoreauGPinLoss(BaseEstimator):
    #setting initial parameters    
    def __init__(self, C = 1, 
                 mu = 0.99, tau_1 = 1, tau_2 = 1, 
                 epsl_1 = 0.25, epsl_2 = 0.5,
                 m=64, max_iter=1000, tol=0.001):
        assert m > 0
        self.m = m
        assert C > 0
        self.C = C
        assert 0 <= mu < 1
        self.mu = mu
        assert tau_1 > 0
        self.tau_1 = tau_1
        assert tau_2 > 0
        self.tau_2 = tau_2
        assert epsl_1 > 0
        self.epsl_1 = epsl_1
        assert epsl_2 > 0
        self.epsl_2 = epsl_2
        assert max_iter > 0 and isinstance(max_iter, int)
        self.max_iter=max_iter
        assert tol > 0 
        self.tol=tol


    ''' cost function (squareHingeloss SVM) '''
    def cost_function(self,w,x,t):
        u = (1-t*np.matmul(x,w))
        p = np.piecewise(u, [
                         u >= (self.epsl_1/self.tau_1) + self.tau_1*self.mu,
                        ((self.epsl_1/self.tau_1) <= u) & (u <= (self.epsl_1/self.tau_1) 
                        + self.tau_1*self.mu),
                        (-(self.epsl_2/self.tau_2) <= u) & (u <= (self.epsl_1/self.tau_1)),
                        (-(self.epsl_2/self.tau_2) - self.tau_2*self.mu <= u) & 
                        (u <= -(self.epsl_2/self.tau_2))
                    ], [
                        lambda u: self.tau_1*(u - (self.epsl_1/self.tau_1))- (((self.tau_1)**2)*self.mu)/2 ,
                        lambda u: (u - (self.epsl_1/self.tau_1))**2/(2*self.mu),
                        0,
                        lambda u: ((u + (self.epsl_2/self.tau_2))**2)/(2 * self.mu),
                        lambda u:   -self.tau_2*(u + (self.epsl_2/self.tau_2))-(((self.tau_2)**2)*self.mu)/2
                    ])
        
        loss = self.C*(np.mean(p))
        cost = 1/2 * np.linalg.norm(w)**2  + loss
        return cost

    ''' approximate ResNaq_BFGS matrix '''
    def Update_RES_BFGS(self,B, dw, dg, I):
            dg_t =  dg[:, np.newaxis]
               
            Bdw = np.dot(B, dw)
            Bdw = np.dot(B, dw)
            dw_t_B = np.dot(dw, B)
            dwBdw = np.dot(np.dot(dw, B), dw)

            p = dg_t*dg
            u = Bdw[:, np.newaxis] * dw_t_B

            B_new = B + p / np.dot(dg, dw) - u / dwBdw + self.delta * I
            return p, u, B_new
        
    def fit(self, x, t):

        # checks for labels
        self.classes_ = np.unique(t)
        t[t==0] = -1

        #initail variables k, stop, w_0
        k = 1
        update = True
        w = np.zeros(len(x[0])+1)
        w_ = np.zeros(len(x[0])+1)
        

        #set up matrices
        B = np.identity(len(x[0])+1)  #inverse hessian
        I = np.identity(len(x[0])+1)  #identity 
        X = np.ones((len(x),len(x[0])+1))
        X[:,:-1] = x 

        obj_sgdss = []
        iter_sgdss = []

        
        
        while k <=(self.max_iter) and update :

                # compute cost function
                cost = self.cost_function(w,X,t)
                obj_sgdss.append(cost)

                r = np.random.randint(len(X),size=self.m)

                X_ = X[r,:]
                t_ = t[r]

                #compute gradient
                u = (1-t_*np.matmul(X_,w))
                p = np.piecewise(u, [
                    u >= (self.epsl_1/self.tau_1) + self.tau_1*self.mu,
                    ((self.epsl_1/self.tau_1) <= u) & (u <= (self.epsl_1/self.tau_1) 
                    + self.tau_1*self.mu),
                    (-(self.epsl_2/self.tau_2) <= u) & (u <= (self.epsl_1/self.tau_1)),
                    (-(self.epsl_2/self.tau_2) - self.tau_2*self.mu <= u) & 
                    (u <= -(self.epsl_2/self.tau_2))
                ], [
                    self.tau_1,
                    lambda u: (u - (self.epsl_1/self.tau_1))/(self.mu),
                    0,
                    lambda u: (u + (self.epsl_2/self.tau_2))/(self.mu),
                    -self.tau_2
                ])
                #print('grad of loss is', p)
                grad = w - self.C*(np.mean(p*t_*X_.T,axis=1))

                

                if  np.linalg.norm(grad)<=self.tol: 
                    update = False
                    #print('DONE')
                else:

                    #update step
                    self.eta = 1/(k)
                    w -= self.eta*grad
                    k += 1
                    
                    w_ += w/k

                    w = w_
                    

                    iter_sgdss.append(k)
                    
               

                    
        self.final_iter = k
        self._coef = w_[:-1]
        self._intercept = w_[-1]
        self.obj_sgdss = obj_sgdss
        self.iter_sgdss = iter_sgdss
        
    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)




class nonlinearRESmoreauGPinLoss(BaseEstimator):
    #setting initial parameters    
    def __init__(self, C = 1, delta = 0.0001, gamma = 0.0001, gammarbf = 1,
                 epsilon_0 = 0.0001, tau_0 = 0.0001,
                 mu = 0.99, tau_1 = 1, tau_2 = 1, 
                 epsl_1 = 0.25, epsl_2 = 0.5,
                 m=2, max_iter=1000, tol=0.001):
        assert m > 0
        self.m = m
        assert C > 0
        self.C = C
        assert epsilon_0 > 0
        self.epsilon_0 = epsilon_0
        assert tau_0 > 0
        self.tau_0 = tau_0
        assert delta > 0
        self.delta = delta
        assert gamma > 0
        self.gamma = gamma
        assert gammarbf > 0
        self.gammarbf = gammarbf
        assert 0 <= mu
        self.mu = mu
        assert tau_1 > 0
        self.tau_1 = tau_1
        assert tau_2 > 0
        self.tau_2 = tau_2
        assert epsl_1 > 0
        self.epsl_1 = epsl_1
        assert epsl_2 > 0
        self.epsl_2 = epsl_2
        assert max_iter > 0 and isinstance(max_iter, int)
        self.max_iter=max_iter
        assert tol > 0 
        self.tol=tol


    ''' approximate ResNaq_BFGS matrix '''
    def Update_RES_BFGS(self,B, dw, dg, I):
            dg_t =  dg[:, np.newaxis]
               
            Bdw = np.dot(B, dw)
            Bdw = np.dot(B, dw)
            dw_t_B = np.dot(dw, B)
            dwBdw = np.dot(np.dot(dw, B), dw)

            p = dg_t*dg
            u = Bdw[:, np.newaxis] * dw_t_B

            B_new = B + p / np.dot(dg, dw) - u / dwBdw + self.delta * I
            return p, u, B_new

    def kernel(self,X,Y):
        X_norm = np.sum(X ** 2, axis = -1)
        Y_norm = np.sum(Y ** 2, axis = -1)
        return ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(X, Y.T),
                'g' : self.gammarbf,
        })
    
        
    def fit(self, x, t):

        # checks for labels
        self.classes_ = np.unique(t)
        t[t==0] = -1

        #collect x for predictive step
        self.x = x
        self.t = t

        #initail variables k, stop, w_0
        k = 1
        update = True
        w = np.zeros(len(x)+1)
        

        #set up matrices
        B = np.identity(len(x)+1)  #inverse hessian
        I = np.identity(len(x)+1)  #identity 
        #X = np.ones((len(x),len(x[0])))
        X = x
     
        
        while k <=(self.max_iter) and update :


                r = np.random.randint(len(X),size=self.m)

                X_ = X[r,:]
                t_ = t[r]
                #print('t=', t_ )

                #map (x_random, X) with rbf kernel
                K = np.ones((self.m,len(x)+1))
                K[:,:-1] = self.kernel(X_,x)
                print('size K', K[:,:-1].shape)

                

                #compute gradient
                u = (1-t_*np.matmul(K,w))
                p = np.piecewise(u, [
                    u >= (self.epsl_1/self.tau_1) + self.tau_1*self.mu,
                    ((self.epsl_1/self.tau_1) <= u) & (u <= (self.epsl_1/self.tau_1) 
                    + self.tau_1*self.mu),
                    (-(self.epsl_2/self.tau_2) <= u) & (u <= (self.epsl_1/self.tau_1)),
                    (-(self.epsl_2/self.tau_2) - self.tau_2*self.mu <= u) & 
                    (u <= -(self.epsl_2/self.tau_2))
                ], [
                    self.tau_1,
                    lambda u: (u - (self.epsl_1/self.tau_1))/(self.mu),
                    0,
                    lambda u: (u + (self.epsl_2/self.tau_2))/(self.mu),
                    -self.tau_2
                ])
                #print('grad of loss is', p)
                grad = w - self.C*(np.mean(p*t_*K.T,axis=1))

                if  np.linalg.norm(grad)<=self.tol:
                    print('grad = ', np.linalg.norm(grad)) 
                    update = False

                else:
                    
                    # update hessian matrix
                    H = np.linalg.inv(B)
                    H_RES = H + self.gamma*I

                    #self.eta = 1/(k+1)
                    self.eta =((self.epsilon_0) * (self.tau_0))/((self.tau_0)+k)
                    # update weight
                    w_new = w - self.eta * np.matmul(H_RES,grad) 



                    #compute new gradient
                    u = (1-t_*np.matmul(K,w_new))
                    p = np.piecewise(u, [
                        u >= (self.epsl_1/self.tau_1) + self.tau_1*self.mu,
                        ((self.epsl_1/self.tau_1) <= u) & (u <= (self.epsl_1/self.tau_1) 
                        + self.tau_1*self.mu),
                        (-(self.epsl_2/self.tau_2) <= u) & (u <= (self.epsl_1/self.tau_1)),
                        (-(self.epsl_2/self.tau_2) - self.tau_2*self.mu <= u) & 
                        (u <= -(self.epsl_2/self.tau_2))
                    ], [
                        self.tau_1,
                        lambda u: (u - (self.epsl_1/self.tau_1))/(self.mu),
                        0,
                        lambda u: (u + (self.epsl_2/self.tau_2))/(self.mu),
                        -self.tau_2
                    ])
                    #print('grad of loss is', p)
                    grad_new = w_new - self.C*(np.mean(p*t_*K.T,axis=1))

                    # get difference of values
                    dw = w_new - w 
                    # get difference of gradients
                    dg = grad_new - grad - (self.delta * dw)

                    _, _, B_new = self.Update_RES_BFGS(B, dw, dg, I)

                    # update step
                    B = B_new
                    w = w_new


                    k += 1
                   
                    
                    
        self.final_iter = k
        self._coef = w[:-1]
        self._intercept = w[-1]


    def predict(self, z):
        X_norm = np.sum(z ** 2, axis = -1)
        Y_norm = np.sum(self.x ** 2, axis = -1)
        x = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(z, self.x.T),
                'g' : self.gammarbf,
        })
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)


class nonlinearSGDGPinLoss(BaseEstimator):
    #setting initial parameters
    def __init__(self, gammarbf = 1,
                    C=0.1, tau_1=1, epsl_1=1, 
                    tau_2=1, epsl_2=1,
                    eta=0.5, max_iter=10000, 
                    tol=0.0001, m=32):
        self.gammarbf = gammarbf
        self.C= C
        self.tau_1=tau_1
        self.epsl_1=epsl_1
        self.tau_2=tau_2
        self.epsl_2=epsl_2
        self.eta=eta
        self.max_iter=max_iter
        self.tol=tol
        self.m=m

    def kernel(self,X,Y):
        X_norm = np.sum(X ** 2, axis = -1)
        Y_norm = np.sum(Y ** 2, axis = -1)
        return ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(X, Y.T),
                'g' : self.gammarbf,
        })

    def fit(self, x, t):
        #collect x for predictive step
        self.x = x
        self.t = t

        #initail variables k, stop, w_0
        k = 1
        update = True
        w = np.zeros(len(x)+1)
        w_T = np.zeros(len(x)+1)


        #set up matrices
        X = x
        


        #solve the positive class problem
        while update and k <= (self.max_iter):
            #stochastic random a datum
            r = np.random.randint(len(X),size=self.m)

            X_ = X[r,:]
            t_ = t[r]

            #map (x_random, X) with rbf kernel
            K = np.ones((self.m,len(x)+1))
            K[:,:-1] = self.kernel(X_,x)
            print('size K', K[:,:-1].shape)


            #compute gradient (change this if using differe loss function)
            
            u = 1-t_*np.matmul(K,w)  #เอา w.T ออก
            p = np.piecewise(u, 
                             [u>self.epsl_1/self.tau_1,
                              (-self.epsl_2/self.tau_2 <= u) & (u<=self.epsl_1/self.tau_1),
                              u<-self.epsl_2/self.tau_2], 
                             [self.tau_1,
                              0,
                              -self.tau_2])
            grad = w - self.C*(np.mean(p*t_*K.T,axis=1))

            #check stopping criterion
           
            if np.linalg.norm(grad) <=self.tol: 
                update = False
                
            else:
                #update step
                self.eta = 1/k
                w -= self.eta*grad
                k += 1

                #collect output from each iteration
                w_T += w
    

        #keep the final solutions
        self.final_iter = k
        self._coef = w_T[:-1]/k
        self._intercept = w_T[-1]/k
        return self


    def predict(self, z):
        X_norm = np.sum(z ** 2, axis = -1)
        Y_norm = np.sum(self.x ** 2, axis = -1)
        x = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(z, self.x.T),
                'g' : self.gammarbf,
        })
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)


class nonlinearSGDhingeLoss(BaseEstimator):
    #setting initial parameters
    def __init__(self, gammarbf = 1,
                    C=0.1,
                    eta=0.5, max_iter=10000, 
                    tol=0.0001, m=32):
        self.gammarbf = gammarbf
        self.C= C
        self.eta=eta
        self.max_iter=max_iter
        self.tol=tol
        self.m=m

    def kernel(self,X,Y):
        X_norm = np.sum(X ** 2, axis = -1)
        Y_norm = np.sum(Y ** 2, axis = -1)
        return ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(X, Y.T),
                'g' : self.gammarbf,
        })

    def fit(self, x, t):
        #collect x for predictive step
        self.x = x
        self.t = t

        #initail variables k, stop, w_0
        k = 1
        update = True
        w = np.zeros(len(x)+1)
        w_T = np.zeros(len(x)+1)


        #set up matrices
        X = x

        #solve the positive class problem
        while update and k <= (self.max_iter):
            #stochastic random a datum
            r = np.random.randint(len(X),size=self.m)

            X_ = X[r,:]
            t_ = t[r]

            #map (x_random, X) with rbf kernel
            K = np.ones((self.m,len(x)+1))
            K[:,:-1] = self.kernel(X_,x)


            #compute gradient (change this if using differe loss function)
            
            u = 1-t_*np.matmul(K,w)  #เอา w.T ออก
            p = np.sign(u)
            p[p==-1] = 0
            grad = w - self.C*(np.mean(p*t_*K.T,axis=1))

            #check stopping criterion
           
            if np.linalg.norm(grad) <=self.tol: 
                update = False
                
            else:
                #update step
                self.eta = 1/k
                w -= self.eta*grad
                k += 1

                #collect output from each iteration
                w_T += w
       

        #keep the final solutions
        self.final_iter = k
        self._coef = w_T[:-1]/k
        self._intercept = w_T[-1]/k
        return self


    def predict(self, z):
        X_norm = np.sum(z ** 2, axis = -1)
        Y_norm = np.sum(self.x ** 2, axis = -1)
        x = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(z, self.x.T),
                'g' : self.gammarbf,
        })
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)