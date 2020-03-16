# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:53:32 2020

@author: Vivek Rathi
"""
import numpy as np
import utils
from matplotlib import pyplot as plt
from scipy import special

# get log pdf for Normal Distribution
def log_pdf(x_i,mu,sigma):
    (sign,logdet) = np.linalg.slogdet(sigma)
    term_sigma = -0.5 * sign * logdet
    ln_exp = (-0.5* np.dot(np.dot((x_i - mu),np.linalg.pinv(sigma)),(x_i - mu)))
    ln_p_xi = ln_exp + term_sigma
    return ln_p_xi

# multi variate log pdf
def mult_var_log_pdf(Test_Data,mu_hat_f,sigma_hat_f):
    pdf = [log_pdf(x_i,mu_hat_f,sigma_hat_f) for x_i in Test_Data]
    return np.array(pdf)


# get data
Train_Face, Train_Non_Face, Test_Face, Test_Non_Face = utils.load_data()

'''
one can take mean as any of the sample image
sigma_mat should be symmetric, one can take diagonal matrix too

'''
# fit gaussian mixture model
def gmm(imgs,K):
    # initialize
    vector_size = 100 # 10*10 images
    w = np.ones(K)/K # weights
    #w = np.random.rand(K)
    mean_vec = np.zeros((K,vector_size)) # array of mean vectors
    for i in range(K):
        mean_vec[i] = imgs[i+K] # means
        
    sigma_mat = np.zeros((K,vector_size,vector_size)) # array of sigma matrices
    sigma_mat[:,:] = np.diag(np.diag(np.cov(np.matrix(imgs),rowvar=False)))
    
    iterations = 30
    L_prev = 0
    L=0
    
    l_k = np.zeros((K,len(imgs)))
    p_k = np.zeros((K,len(imgs)))
    r_k = np.zeros((K,len(imgs)))
    pdf_k = np.zeros((K,len(imgs)))
    
    print("Initialization")
    print(w)
    for c in range(iterations):
        L_prev = L
        
        # E Step; get latent variable     
        for i in range(K):
            pdf_k[i] = mult_var_log_pdf(imgs,mean_vec[i],sigma_mat[i])
            print("In E Step")
            l_k[i] = np.log(w[i])+pdf_k[i]
        
    #   got latent variable
        r_k = special.softmax(l_k,axis=0)
    
        print("E Step for iteration-{} done".format(c))
        
    #    # M Step; get estimates
        sum_r_k_i = np.sum(r_k,axis=1)
        sum_r_k_k = np.sum(sum_r_k_i)
        for i in range(K):
            mean_vec[i]=np.dot(r_k[i],imgs) / sum_r_k_i[i]
            sigma_m = np.dot(r_k[i],np.square(imgs-mean_vec[i]))
            sigma_mat[i] = np.diag(sigma_m) / sum_r_k_i[i]
            w[i] = sum_r_k_i[i]/sum_r_k_k
        
        print("M Step for iteration-{} done".format(c))
        print(w)
        
    #    # stop condition
       
        for i in range(K):
            pdf_k[i] = mult_var_log_pdf(imgs,mean_vec[i],sigma_mat[i])
            p_k[i] = np.log(w[i]) + pdf_k[i]
            
        p_k_sum = np.sum(np.exp(p_k),axis=0)
        L = np.sum(np.log(p_k_sum))
        print("Iteration-{} Completed, L ={}".format(c,L))
        
        # check for convergence
        diff = round(L-L_prev,2)
        if (abs(diff) <= 0.01):
            break

    return w,mean_vec,sigma_mat

# visualize means & covariances
def visualize(m1,sigma1):
    m1_mg = m1[0].reshape(10,10)
    m2_mg = m1[1].reshape(10,10)
    m3_mg = m1[2].reshape(10,10)
    s1 = (np.diag(sigma1[0]).reshape(10,10))
    sigma1_img = s1/s1.max()
    s2 = (np.diag(sigma1[1]).reshape(10,10))
    sigma2_img = s2/s2.max()
    s3 = (np.diag(sigma1[2]).reshape(10,10))
    sigma3_img = s3/s3.max()
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(231)
    plt.imshow(m1_mg,cmap='gray')
    plt.title('Mean K=1')
    plt.subplot(232)
    plt.imshow(m2_mg,cmap='gray')
    plt.title('Mean K=2')
    plt.subplot(233)
    plt.imshow(m3_mg,cmap='gray')
    plt.title('Mean K=3')
    plt.subplot(234)
    plt.imshow(sigma1_img,cmap='gray')
    plt.title('Covar K=1')
    plt.subplot(235)
    plt.imshow(sigma2_img,cmap='gray')
    plt.title('Covar K=2')
    plt.subplot(236)
    plt.imshow(sigma3_img,cmap='gray')
    plt.title('Covar K=3')

# fit gmm model
def fit_gmm(w,mean,sigma,data,K):
    pdf_k = np.zeros((K,len(data)))
    p_k = np.zeros((K,len(data)))
    for i in range(K):
        pdf_k[i] = mult_var_log_pdf(data,mean[i],sigma[i])
#        p_k[i] = w[i] * pdf_k[i]
        p_k[i] = np.log(w[i]) + pdf_k[i]
#        plt.figure()
#        plt.hist(p_k,label='pf')
#        plt.show()
    
    p = np.sum(p_k,axis=0)
    return p,p_k


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
        false_pos[i] = np.sum(pn >= thresh[i]) / 100
        false_neg[i] = np.sum(pf < thresh[i]) / 100
        true_pos[i] = 1 - false_neg[i]
    plt.figure()
    plt.plot(false_pos, true_pos)
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
# get weights, mean, sigma for gmm model
w_f,mean_vec_f,sigma_mat_f = gmm(Train_Face,3)
w_nf,mean_vec_nf,sigma_mat_nf = gmm(Train_Non_Face,3)

# visualize mean & covariance
visualize(mean_vec_f,sigma_mat_f)
visualize(mean_vec_nf,sigma_mat_nf)



# get the log likelihoods
_,pf1 = fit_gmm(w_f,mean_vec_f,sigma_mat_f,Test_Face,3)
_,pf0 = fit_gmm(w_nf,mean_vec_nf,sigma_mat_nf,Test_Face,3)

_,pn1 = fit_gmm(w_f,mean_vec_f,sigma_mat_f,Test_Non_Face,3)
_,pn0 = fit_gmm(w_nf,mean_vec_nf,sigma_mat_nf,Test_Non_Face,3)
    
# get the likelihoods
pf1 = (np.exp(pf1)).sum(axis=0)
pf0 = (np.exp(pf0)).sum(axis=0)
pn1 = (np.exp(pn1)).sum(axis=0)
pn0 = (np.exp(pn0)).sum(axis=0)

# get the posteriors
post_1f = [(p1)/(p1+p2) for p1,p2 in zip(pf1,pf0)] # true positive
post_0f = [(p2)/(p1+p2) for p1,p2 in zip(pf1,pf0)]

post_1n = [(p1)/(p1+p2) for p1,p2 in zip(pn1,pn0)] # false positive
post_0n = [(p2)/(p1+p2) for p1,p2 in zip(pn1,pn0)]

# confusion matrix
p1f = np.array(post_1f)
p1n = np.array(post_1n)

false_pos = np.sum(p1n > 0.5)
FPR = false_pos/100
false_neg = np.sum(p1f < 0.5)
FNR = false_neg/100
MSR = (false_pos + false_neg)/200


print("FPR:{}".format(FPR))
print("FNR:{}".format(FNR))
print("MSR:{}".format(MSR))

# ROC plot
ROC(post_1f,post_1n)



    
    
