# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 02:15:04 2020

@author: Vivek Rathi
"""

# import APIs
import numpy as np # array calculations
import utils # to load data
from matplotlib import pyplot as plt  # for plotting
from scipy.special import digamma, gammaln # for digamma and gammaln
from scipy.optimize import fminbound # for fminbound (line search)

# get the data
Train_Face, Train_Non_Face, Test_Face, Test_Non_Face = utils.load_data()

# get gaussian parameters, mu & sigma
def gaussian_param(list_img):
    mu_hat = np.mean(list_img,axis=0)
    sigma_hat = np.cov(np.matrix(list_img),rowvar=False)
#    sigma_hat = np.cov(np.transpose(list_img))
    sigma_hat = np.diag(np.diag(sigma_hat))
    return mu_hat, sigma_hat

# cost function
# this will be input to fminbound
def tcost(v,E_h,E_log_h):
    c = np.zeros(len(E_h))
    for i in range(len(E_h)):
        c[i] = (v/2)*np.log(v/2) + gammaln(v/2) - ((v/2)-1)*E_log_h[i] + (v/2)*E_h[i]
        
    return -1*np.sum(c)
    
# my cost_function with argmin function
def my_tcost(v,E_h,E_log_h,imgs):
    t_sum = np.zeros(1000)
    c = np.zeros((len(imgs)))
    for v in range(1,1000):
        for i in range(len(imgs)):
            c[i] = (v/2)*np.log(v/2) + gammaln(v/2) - ((v/2)-1)*E_log_h[i] + (v/2)*E_h[i]
        t_sum[v-1] = -1*np.sum(c)
    return np.argmin(t_sum) + 1
    
#fit t distribution
def fit_t(imgs):
    v = 1000  # DOF
    I = len(imgs) # Dimension
    mean,sigma = gaussian_param(imgs)
    #    imgs = np.matrix(imgs)
    D = imgs[0].shape[0]
    delta = np.zeros(I)
    E_h = np.zeros(I)
    E_log_h = np.zeros(I)
    L = 0
    L_prev = 0
    for t in range(10):
        
        L_prev  = L
        # E Step - get hidden variables
        print("E Step Start")
        for i in range(len(imgs)):
            delta[i] = np.dot(np.dot((imgs[i]-mean),np.linalg.pinv(sigma)),(imgs[i]-mean))
            E_h[i] = (v+D)/(v+delta[i])
            E_log_h[i] = digamma((v + D) / 2) - np.log((v + delta[i]) / 2)
        
        
        print("E Step End")
        
        # M Step - get mean & sigma
        print("M Step Start-{}".format(t))
        
        E_h_sum = E_h.sum()
        mean = np.divide(np.dot(E_h,imgs),E_h_sum)
#        print(mean)
        sigma = np.divide(np.dot(E_h,np.square(imgs-mean)),E_h_sum)
        sigma = np.diag(sigma)
        v = fminbound(tcost,0,500,args=(E_h,E_log_h))
#        v = my_tcost(v,E_h,E_log_h,imgs)
        
#        print(v)
        
    
        
        print("Update delta")
        for i in range(len(imgs)):
            delta[i] = np.dot(np.dot((imgs[i]-mean),np.linalg.pinv(sigma)),(imgs[i]-mean))
        
        # get log likelihood
        _, logdet = np.linalg.slogdet(sigma)
        L = I * gammaln((v+D)/2) - (I*D*np.log(v*2*np.pi))/2 -I*logdet/2 -I*gammaln(v/2)
        print(L)
        
        # check for convergence
        diff = round(L-L_prev,2)
        if (abs(diff) <= 0.01):
            break
    
    return mean,sigma,v

# predict t distribution
def predict_t(imgs,v,mean,sigma):
    D = len(mean)
    _,logdet = np.linalg.slogdet(sigma)
    pdf = []
    for i in range(len(imgs)):
        P = gammaln((v+D)/2) - ((D/2)*np.log(v*np.pi) + 0.5 * logdet + gammaln(v/2)) -((v+D)/2) * np.log(1 + np.dot(np.dot((imgs[i]-mean),np.linalg.pinv(sigma)),(imgs[i]-mean))/v)
        pdf.append(P)
    return pdf

# ROC plot
def ROC(p1f,p1n):
    pf = np.array(p1f)
    pn = np.array(p1n)
    false_pos = np.zeros(1000)
    false_neg = np.zeros(1000)
    true_pos = np.zeros(1000)
    thresh = np.arange(1000)/999
    thresh = thresh[::-1]
    
    for i in range(0, 1000):
        false_pos[i] = np.sum(pn < thresh[i]) / 100
        false_neg[i] = np.sum(pf >= thresh[i]) / 100
        true_pos[i] = 1 - false_neg[i]
    plt.figure()
    plt.plot(false_pos, true_pos)
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
# visualize mean & covariance
def visualize(m1,sigma1,type_):
    m1_mg = m1.reshape(10,10)
    s1 = (np.diag(sigma1).reshape(10,10))
    sigma1_img = s1/s1.max()

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(m1_mg,cmap='gray')
    plt.title('Mean {}'.format(type_))
    plt.subplot(122)
    plt.imshow(sigma1_img,cmap='gray')
    plt.title('Covariance {}'.format(type_))


# fit the t distribution to the data
mean_f,sigma_f,v_f = fit_t(Train_Face)
mean_nf,sigma_nf,v_nf = fit_t(Train_Non_Face)


# visualize mean & covariance
visualize(mean_f,sigma_f,"Face")
visualize(mean_nf,sigma_nf,"Non Face")


#v_f = 10
#v_nf = 15
# get the likelihoods
pf1 = predict_t(Test_Face,v_f,mean_f,sigma_f)
pf0 = predict_t(Test_Face,v_nf,mean_nf,sigma_nf)
pn1 = predict_t(Test_Non_Face,v_f,mean_f,sigma_f)
pn0 = predict_t(Test_Non_Face,v_nf,mean_nf,sigma_nf)

#plt.figure()
#plt.hist(pf1+pf0,label='pf')
#plt.hist(pn0+pn1,label='pnf')
#plt.legend(loc='upper right')
#plt.show()

# get the posteriors
post_1f = [(p1)/(p1+p2) for p1,p2 in zip(pf1,pf0)] # true positive
post_0f = [(p2)/(p1+p2) for p1,p2 in zip(pf1,pf0)]

post_1n = [(p1)/(p1+p2) for p1,p2 in zip(pn1,pn0)] # false positive
post_0n = [(p2)/(p1+p2) for p1,p2 in zip(pn1,pn0)]


plt.figure()
plt.hist(post_1f,label='pf')
plt.hist(post_1n,label='pnf')
plt.legend(loc='upper right')
plt.show()

# get confusion matrix
p1f = np.array(post_1f)
p1n = np.array(post_1n)

false_pos = np.sum(p1n < 0.5)
FPR = false_pos/100
false_neg = np.sum(p1f > 0.5)
FNR = false_neg/100
MSR = (false_pos + false_neg)/200

print("FPR:{}".format(FPR))
print("FNR:{}".format(FNR))
print("MSR:{}".format(MSR))

# ROC plot
ROC(post_1f,post_1n)












    
    
    

