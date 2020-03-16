# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:58:29 2020

@author: Vivek Rathi

float images - pdf positive
int images - pdf negative
swap the threshold if you are using any

now using int
"""
import numpy as np
import utils
from matplotlib import pyplot as plt

#get data 
Train_Face, Train_Non_Face, Test_Face, Test_Non_Face = utils.load_data()


# range calulation
def rang(list_):
    return (min(list_),max(list_))

# get gaussian parameters, mu & sigma
def gaussian_param(list_img):
    mu_hat = np.mean(list_img,axis=0)
    sigma_hat = np.cov(np.matrix(list_img),rowvar=False)
#    sigma_hat = np.cov(np.transpose(list_img))
    sigma_hat = np.diag(np.diag(sigma_hat))
    return mu_hat, sigma_hat

# get p(x|w) i.e. likelihood from pdf
def gaus_pdf_multi_N_D(x_i,mu,sigma):
    d = len(x_i)
    scalar = 1/((2*np.pi)**(d/2) * np.linalg.det(sigma)**(1/2))
    exp = np.exp(-0.5* np.dot(np.dot((x_i - mu),np.linalg.pinv(sigma)),(x_i - mu)))
    p_xi = scalar * exp
    return p_xi

# get log pdf
def log_pdf(x_i,mu,sigma):
    d = len(x_i)
    (sign,logdet) = np.linalg.slogdet(sigma)
    term_sigma = -0.5 * sign * logdet
    ln_exp = (-0.5* np.dot(np.dot((x_i - mu),np.linalg.pinv(sigma)),(x_i - mu)))
    ln_p_xi = ln_exp + term_sigma
    return ln_p_xi

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
def visualize(m1,sigma1,m2,sigma2):
    mu_f_img = m1.reshape(10,10)
    mu_nf_img = m2.reshape(10,10)
    sigma1 = (np.diag(sigma1).reshape(10,10))
    sigma1 = sigma1/sigma1.max()
    sigma2 = (np.diag(sigma2).reshape(10,10))
    sigma2 = sigma2/sigma2.max()
    
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.imshow(mu_f_img,cmap='gray')
    plt.title('Mean For Face')
    plt.subplot(222)
    plt.imshow(sigma1,cmap='gray')
    plt.title('Covariance For Face')
    plt.subplot(223)
    plt.imshow(mu_nf_img,cmap='gray')
    plt.title('Mean For Non Face')
    plt.subplot(224)
    plt.imshow(sigma2,cmap='gray')
    plt.title('Covariance For Non Face')
    

# get the MLE estimates for training data i.e. fit the data
mu_hat_f, sigma_hat_f = gaussian_param(Train_Face)
mu_hat_nf, sigma_hat_nf = gaussian_param(Train_Non_Face)

# visualize mean & covariance
visualize(mu_hat_f,sigma_hat_f,mu_hat_nf,sigma_hat_nf)

# get the likelihoods
pf1 = [log_pdf(x_i,mu_hat_f,sigma_hat_f) for x_i in Test_Face]
pf0 = [log_pdf(x_i,mu_hat_nf,sigma_hat_nf) for x_i in Test_Face]

pn1 = [log_pdf(x_i,mu_hat_f,sigma_hat_f) for x_i in Test_Non_Face]
pn0 = [log_pdf(x_i,mu_hat_nf,sigma_hat_nf) for x_i in Test_Non_Face]


# get posteriors
post_1f = [(p1)/(p1+p2) for p1,p2 in zip(pf1,pf0)]
post_0f = [(p2)/(p1+p2) for p1,p2 in zip(pf1,pf0)]

post_1n = [(p1)/(p1+p2) for p1,p2 in zip(pn1,pn0)]
post_0n = [(p2)/(p1+p2) for p1,p2 in zip(pn1,pn0)]


plt.figure()
plt.hist(post_1f,label='pf')
plt.hist(post_1n,label='pnf')
plt.legend(loc='upper right')
plt.show()

# confusion matrix
p1f = np.array(post_1f)
p1n = np.array(post_1n)

false_pos = np.sum(p1n < 0.5)
FPR = false_pos/100
false_neg = np.sum(p1f > 0.5)
FNR = false_neg/100
MSR = (false_pos + false_neg)/200
#
print("FPR:{}".format(FPR))
print("FNR:{}".format(FNR))
print("MSR:{}".format(MSR))

# ROC plot
ROC(post_1f,post_1n)

