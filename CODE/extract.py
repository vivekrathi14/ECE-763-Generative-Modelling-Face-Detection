# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:04:24 2020

@author: Vivek Rathi
"""

import os
import cv2


# function to deal with if rectangle goes past the boundaries of the image
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
proj_dir = 'E:\Project01'

# merge all annotation into one single file and add path for it
input_ellipse = os.path.join(proj_dir,'FDDB-folds', 'Final','ellipse_all.txt')


# create directories for Extracted Images
make_dir(os.path.join(proj_dir,'Extracted-Face'))
make_dir(os.path.join(proj_dir,'Extracted-NonFace'))

# get list of elements of the text file, much like csv file
with open(input_ellipse) as annotation:
    data = [line.rstrip('\n') for line in annotation]

total_faces = 0
i = 0
crop_size=(20,20)
# go line by line and import the data according to the readme provided by FDDB
while i < len(data):
    img_file = data[i] + '.jpg'
    img_path = os.path.join(proj_dir,'originalPics',img_file)
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    num_faces = int(data[i + 1])
    # count running total of number of faces in images
    total_faces = total_faces + num_faces
    # for each face in the image, import the annotations
    for j in range(num_faces):
        face = data[i + 2 + j].split()[0:6]
        r_major = float(face[0])
        r_minor = float(face[1])
        angle = float(face[2])
        c_x = float(face[3])
        c_y = float(face[4])
        
        # face images
        img_patch_face = img[int(c_y-r_major):int(c_y+r_major),int(c_x-r_minor):int(c_x+r_minor),:]
        if img_patch_face.shape[0] == 0 or img_patch_face.shape[1] == 0:
            continue
        else:
            resized_face = cv2.resize(img_patch_face, crop_size, interpolation = cv2.INTER_AREA)
        
            img_f_file = 'Face-Image-' + str(i+j) + '.jpg'
            img_path_f = os.path.join(proj_dir,'Extracted-Face', img_f_file)
            cv2.imwrite(img_path_f, resized_face)
        # non-face images
        
        img_patch_non_f = img[int(c_y+r_major+5):h,int(c_x+r_minor+5):w,:]
        if img_patch_non_f.shape[0] == 0 or img_patch_non_f.shape[1] == 0:
            continue
        else:
            resized_nf = cv2.resize(img_patch_non_f, crop_size, interpolation = cv2.INTER_AREA)
            img_nf_file = 'Non-Face-Image-' + str(i+j) + '.jpg'
            img_path_nf = os.path.join(proj_dir,'Extracted-NonFace', img_nf_file)
            cv2.imwrite(img_path_nf, resized_nf)
        
        
        
    i = i + num_faces + 2
    