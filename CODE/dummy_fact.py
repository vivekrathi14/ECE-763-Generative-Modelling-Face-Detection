# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:10:20 2020

@author: Vivek Rathi
"""


# import APIs
import numpy as np
import utils # to get data
from matplotlib import pyplot as plt # for plotting
from scipy.special import logsumexp # for logsumexp

Train_Face, Train_Non_Face, Test_Face, Test_Non_Face = utils.load_data()


K = 5 # factors

def gaussian_param(list_img):
    mu_hat = np.mean(list_img,axis=0)
    sigma_hat = np.cov(np.matrix(list_img),rowvar=False)
    sigma_hat = np.diag(np.diag(sigma_hat))
    return mu_hat, sigma_hat

#Initialization
def fit_fact(imgs):
    I = len(imgs)
    D = imgs[0].shape[0]
    mean,sigma = gaussian_param(imgs)
#    phi = np.zeros((D,K))
#    phi = np.ones((D,K)) * 50 # gives accuarcay of 75%
#    phi = np.random.rand(D,K) # varaible accuraccy
#    phi = np.random.uniform(0.0,1,(D,K)) # givers accuraccy of 78%
    phi = np.random.uniform(50,80,(D,K)) # good accurcay
#    phi = sigma[:,:K] # gives decent accuracy of 80%
    E_h = np.zeros((I,K,1))
    E_h_hT = np.zeros((I,K,K))
    L_prev = 0
    L = 0
    for c in range(10):
        # E Step - get hidden variables
        L_prev = L
        print("E Step")
        for i in range(I):
            sigma_inv = np.linalg.pinv(sigma)
            t1 = np.dot(np.dot(phi.T,sigma_inv),phi) + 1
            t1_inv = np.linalg.pinv(t1)
            E_h[i] = np.dot(np.dot(np.dot(t1_inv,phi.T),sigma_inv),(imgs[i]-mean)).reshape(K,1)
            E_h_hT[i] = t1_inv + np.dot(E_h[i],E_h[i].T)
            
        # M Step - get phi and sigma
        print("M Step")
        #phi
        t2_phi = np.linalg.pinv(E_h_hT.sum(axis=0))
        t1_phi = 0
        
        for i in range(I):
            t1_phi = t1_phi + np.dot((imgs[i]-mean).reshape(D,1),E_h[i].T)
        
        phi = np.dot(t1_phi,t2_phi)
        
#        print(phi)
        # sigma
        b = np.dot(phi,E_h[:])
        term2 = np.dot(b[:,:,0],imgs-mean)
        term1 = np.dot(np.transpose(imgs-mean),imgs-mean)
        sigma = term1 - term2
        sigma = np.diag(np.diag(sigma))/I
        print("Iteration - {}".format(c))
        
        sigma_d = np.dot(phi,phi.T) + sigma
        
        # Get likelihood
        p = [log_pdf(x_i,mean,sigma_d) for x_i in imgs]
        # as we have log likelihood use logsumexp to get exact pdf sum
        L = logsumexp(p)
        print(L)
        
        # check for convergence
        diff = round(L-L_prev,2)
        if (abs(diff) <= 0.01):
            break
        
        
        
        
    # update sigma  
    sigma = np.dot(phi,phi.T) + sigma
    
    # return mean & sigma
    return mean,sigma


# get logpdf for given normal distribution
def log_pdf(x_i,mu,sigma):
    (sign,logdet) = np.linalg.slogdet(sigma)
    term_sigma = -0.5 * logdet
    ln_exp = (-0.5* np.dot(np.dot((x_i - mu),np.linalg.pinv(sigma)),(x_i - mu)))
    ln_p_xi = ln_exp + term_sigma
    return ln_p_xi


# visualise mean and convariance(diagonal elements)
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
    
# fit the data for t distribution
mean_f,sigma_f = fit_fact(Train_Face)
mean_nf,sigma_nf = fit_fact(Train_Non_Face)

# visualize mean and covariance
visualize(mean_f,sigma_f,"Face")
visualize(mean_nf,sigma_nf,"Non Face")


# calaculate log_pdfs
pf1 = [log_pdf(x_i,mean_f,sigma_f) for x_i in Test_Face]
pf0 = [log_pdf(x_i,mean_nf,sigma_nf) for x_i in Test_Face]

pn1 = [log_pdf(x_i,mean_f,sigma_f) for x_i in Test_Non_Face]
pn0 = [log_pdf(x_i,mean_nf,sigma_nf) for x_i in Test_Non_Face]

# calculate posteriors
post_1f = [(p1)/(p1+p2) for p1,p2 in zip(pf1,pf0)] # true positive
post_0f = [(p2)/(p1+p2) for p1,p2 in zip(pf1,pf0)]

post_1n = [(p1)/(p1+p2) for p1,p2 in zip(pn1,pn0)] # false positive
post_0n = [(p2)/(p1+p2) for p1,p2 in zip(pn1,pn0)]

# plot true positives & false positives
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

# plot ROC
ROC(post_1f,post_1n)

