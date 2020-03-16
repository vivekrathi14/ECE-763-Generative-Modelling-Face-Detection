# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:19:03 2020

@author: Vivek Rathi
"""

import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

# get list of input images
def create_list_images(path):
    List_images = []
    for imagepath in glob.glob(path + '\*'):
        image = cv2.imread(imagepath,0)
#        image = image.reshape(image.shape[0]*image.shape[1],1)
        image = cv2.resize(image, (10,10), interpolation = cv2.INTER_AREA)
        image = image.flatten()
#        image = image/255
        List_images.append(image.astype(np.float32))
    return List_images

# get the data 
def load_data():
    proj_dir = os.getcwd()
    train_f = os.path.join(proj_dir,'Train_face')
    train_nf = os.path.join(proj_dir,'Train_nface')
    test_f = os.path.join(proj_dir,'Test_face')
    test_nf = os.path.join(proj_dir,'Test_nface')
    Train_Face = create_list_images(train_f)
    Train_Non_Face = create_list_images(train_nf)
    Test_Face = create_list_images(test_f)
    Test_Non_Face = create_list_images(test_nf)
    
    return Train_Face, Train_Non_Face, Test_Face, Test_Non_Face
    

#def ROC_plot_inbuilt(pre_f,pred_nf,length):    
#    predictions = np.append(pre_f, pred_nf)
#    actual = np.append([1]*length,[0]*length)
#    false_positive_rate, true_positive_rate, _ = roc_curve(actual, predictions)
##    roc_auc = auc(false_positive_rate, true_positive_rate)
#    plt.figure()
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('ROC Curve')
#    plt.plot(false_positive_rate, true_positive_rate, 'b')
#    plt.show()