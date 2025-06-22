#Importing the required libraries
import os         
import cv2 as cv         
import numpy as np   
import tensorflow as tf
from tensorflow.keras.models import load_model, Model     
from sklearn.cluster import KMeans                       
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image                         
import matplotlib.image as mpimg              
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score        
from tabulate import tabulate     
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def get_files(path_to_files, size):
    """
    Helper function to load and resize images with the appropriate size for the input model. 
    """
    fn_imgs = []
    files = [file for file in os.listdir(path_to_files)]       #Storing all image files names
    for file in files:
        file_path = os.path.join(path_to_files,file)
        image_array = cv.imread(file_path)
        img = cv.resize(image_array, size, interpolation=cv.INTER_AREA) #Reading and resizing each image
        fn_imgs.append([file, img])             #Accumulating each image to a variable          
    return dict(fn_imgs)                        #Return dictionary of file names and images    

def feature_vector(img_arr, model):          
    if img_arr.shape[2] == 1:       
      img_arr = img_arr.repeat(3, axis=2)     #expand dimension if image is in grey scale
    arr4d = np.expand_dims(img_arr, axis=0)   #Expanding image dimension (1, 224, 224, 3) to make it a tensor compatable with keras model
    return model.predict(arr4d)[0,:]          #Returning image features through last layer of your model


#Extracting features from all the images
def feature_vectors(imgs_dict, model):
    f_vect = {}
    for fn, img in imgs_dict.items():         #For loop over each image in the imgs_dict
      f_vect[fn] = feature_vector(img, model) #Calling feature vector function to extract the features
    return f_vect                             #Returning features dictionary
     