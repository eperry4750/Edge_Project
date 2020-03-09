The operating system is Windows 10
64 bit  Anaconda Python 3.7 version is installed with the --full package
The IDE is Spyder

Open the Anaconda command prompt

setup Openvino Open CV enviroment
cd C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\bin
setupvars.bat

Download Models
The frozen_inference_graph.xml model was downlaoded 
from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

The Caffe fullfacedetection.xml model was downloaded from 
from https://github.com/PINTO0309/MobileNet-SSD-RealSense/tree/master/caffemodel/Facedetection

The face-detection-adas-0001.xml is a pretrained Edge model.

Thhe naive model is the OpenCV 'haarcascade_frontalface_default.xml' model downloaded 
from https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81

The Faces 1999(Front) database was downloaded from http://www.vision.caltech.edu/html-files/archive.html

import cv2
import numpy as np
import os
import time
from inference import Network

CPU_EXTENSION = r"C:\Program Files (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"
CPU = "CPU"
CONFIDENCE =  0.975

def dataset():
    '''
    Downloads images and add labels
    '''    
    images = []
    labels = []
    labels_dict = {}
    people = [person for person in os.listdir("faces_database/")]
    for i, person in enumerate(people):
            labels_dict[i] = person
            labels.append(person)
            
    for image in os.listdir("faces_database/"):
            images.append(cv2.imread("faces_database/" + image, 1))
    return images, np.array(labels), labels_dict
    
def face_count(result, confidence, width, height):
    '''
    Draw bounding boxes onto the frame.
    Count number of faces found
    '''
    # get dictionary key
    name =  list(result.keys())[0]
    face_count = 0
    for box in result[name][0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= confidence:
            face_count += 1
    return face_count
    
'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)
    return image
    
def perform_inference(images, model):
    # image = r"C:\Users\eperr\Anaconda3\Scripts3\Edge\Project\images\faces-1.jpg    
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    inference_network = Network()
    n, c, h, w = inference_network.load_model(model, CPU,  CPU_EXTENSION)    
    # Read the input image
    # image = cv2.imread(image)
    ### TODO: Preprocess the input image
    count = 0
    for image in images:
        preprocessed_image = preprocessing(image, h, w)        
        # Perform synchronous inference on the image
        inference_network.sync_inference(preprocessed_image)   
        # Obtain the output of the inference request
        result = inference_network.extract_output()           
        ### TODO: Update the image to include detected bounding boxes
        count += face_count(result, CONFIDENCE, image.shape[1], image.shape[0])
        # print("Found {0} faces!".format(face_count))
    return count
    
def naive_model(images):        
    # face detection image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # image = cv2.imread('images/sitting-on-car.png',0)
    # image = cv2.imread('images/white-face-black-face.jpg', 0)
    scale_factor = 1.3
    count = 0
    for image in images:
        face = face_cascade.detectMultiScale(image, scale_factor, 5)
        count += len(face)
    return count       
images, labels, labels_dict = dataset()

models = [r"C:\Users\eperr\Anaconda3\Scripts3\Edge\Project\frozen_inference_graph.xml",
r"C:\Users\eperr\Anaconda3\Scripts3\Edge\Project\caffe\IR\fullfacedetection.xml",
r"C:\Users\eperr\Anaconda3\Scripts3\Edge\Project\intel\face-detection-adas-0001\FP32\face-detection-adas-0001.xml"]

start_time = time.time()
print("The haarcascade_frontalface_default.xml model found ", naive_model(images))
print("--- in %s seconds ---" % (time.time() - start_time))
print("The confidence level is 97.5% and the database has 450 faces")

model_names = ["The Tensorflow frozen_inference_graph.xml model found ", "The Caffe fullfacedetection.xml model found ",
"The FP32 face-detection-adas-0001.xml model found "]

start_time = time.time()
for index, model in enumerate(models):
    print(model_names[index], perform_inference(images, model))
    print("--- in %s seconds ---" % (time.time() - start_time))
 
    





        
