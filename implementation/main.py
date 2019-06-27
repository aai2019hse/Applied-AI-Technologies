import socket

#Sockets for IPC
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

def send_udp_message(message): 
    sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
    sock.sendto(message, (UDP_IP, UDP_PORT))
    
    pass

send_udp_message(b'BOOTING')

from imutils.video import VideoStream
from imutils.video import FPS

import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D,Lambda, Flatten, Dense

import h5py

from keras import backend as K

import numpy as np

import argparse
import imutils
import time
import cv2

#FUNCTIONS
def get_siamese_model(input_shape):
    """
    Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape, name="input_1")
    right_input = Input(input_shape, name="input_2")
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(12, (10,10), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(24, (7,7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(24, (4,4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (4,4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid', name="main_output")(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

print("[INFO] loading SSD..")
# load ssd
net = cv2.dnn.readNetFromTensorflow('./model/mobilenet_ssd/sorted_inference_graph.pb', './model/mobilenet_ssd/output_new.pbtxt')

print("[INFO] loading Siamese Network.. (this might take a while)")
# read siamese network
siamese_network_model = get_siamese_model((128, 128, 1))
siamese_network_model.load_weights(os.path.join("./model/siamese", "weights.6800.h5"))

print("[INFO] loading Referance Image")
# Load a reference image
image_reference = cv2.imread("./reference/reference_image.jpg", 2)

print("[INFO] setup NCS")
# specify the target device as the Myriad processor on the NCS
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True, resolution=(640, 360)).start()

# Wait for stream
for i in range(1, 5):  
    print("Waiting for Images:" ,(5 - i))
    time.sleep(0.1)

# Program Vars
#ssd_classes = ["cat_face", "cat_face"]
ssd_colors = [[255,0,0],[0,0,255]]
time_catface_detection = 0  
predictions_list = []

# main 
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame_org = vs.read()
    
    #cv2.namedWindow("Camera Image Org")
    #cv2.imshow("Camera Image Org", frame_org)
    
    if(time_catface_detection == 0): 
        send_udp_message(b'RUNNING')
    
    frame = imutils.resize(frame_org, width=300)
    
    frame_org_h, frame_org_w = frame_org.shape[:2]
    
    frame_scaled_h, frame_scaled_w = frame.shape[:2]
    
    scale_width = frame_org_w / frame_scaled_w
    scale_height = frame_org_h / frame_scaled_h
    
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    	
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    
    confidences = detections[0, 0, :, 2]
    if confidences.size > 0:
        index_of_maximum = np.argmax(confidences)
                
        i = 0 #i = index_of_maximum
        
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        if (confidence > 0.75):
            send_udp_message(b'CALC')
            
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame_scaled_w, frame_scaled_h, frame_scaled_w, frame_scaled_h])
            (startX, startY, endX, endY) = box.astype("int")
                 
            # As our SSD has some offset within its detetcions, we need to correct them 
            shift = 10 # shift in pixels
            
            corr_start_x = startX - shift
            corr_start_y = startY - shift
            corr_end_x = endX - shift
            corr_end_y = endY
            
            if(corr_start_x < 0):
                corr_start_x = 0
            if(corr_start_y < 0):
                corr_start_y = 0
                
            cv2.rectangle(frame, (corr_start_x, corr_start_y), (corr_end_x , corr_end_y ), ssd_colors[idx], 2)
            
            start = time.time()
            
            #crop_catface = frame[startY:endY, startX:endX]
            crop_catface = frame_org[int(corr_start_y*scale_height):int(corr_end_y*scale_height), int(corr_start_x*scale_width):int(corr_end_x*scale_width)]
 
            ts_crop = time.time()
            
            # Fill Image and resize to 128 x 128
            (h,w) = crop_catface.shape[:2] 
            if w > h:
                pad_image = np.zeros((w+int(w)%2,w+int(w)%2,3),np.uint8)
                h_offset = int((w-h)/2)
                w_offset = int(0)
                pad_image[h_offset:h_offset+h,w_offset:w_offset+w,:] = crop_catface
            else:
                pad_image = np.zeros((h+int(h)%2,h+int(h)%2,3),np.uint8)
                h_offset = int(0)
                w_offset = int((h-w)/2)
                pad_image[h_offset:h_offset+h,w_offset:w_offset+w,:] = crop_catface
            
            ts_resize = time.time()
            
            pad_image = cv2.cvtColor(pad_image,code=cv2.COLOR_BGR2RGB)
            pad_image = cv2.cvtColor(pad_image,code=cv2.COLOR_BGR2RGB)  
            pad_image = imutils.resize(pad_image,width=128)
            
            ts_cvt= time.time()
            
            (h,w) = pad_image.shape[:2]
            
            siamese_input_8 = np.zeros((128,128,3),np.uint8)
            siamese_input_8 = pad_image[0:h,0:w,:]
            
            siamese_input_32 = np.zeros((128,128),np.uint32)
            siamese_input_32 = np.add(np.multiply(siamese_input_8[:,:,0], 2**16), np.multiply(siamese_input_8[:,:,1], 2**8))
            siamese_input_32 = np.add(siamese_input_32, siamese_input_8[:,:,2])
            
            ts_conv= time.time()
            
            input_h,input_w = siamese_input_32.shape
            if input_h == 128 and input_w == 128:
                pairs = [siamese_input_32.reshape(1, 128, 128, 1), image_reference.reshape(1, 128, 128, 1)]
                prediction = siamese_network_model.predict(pairs)
                
                end = time.time()
                
                if(time_catface_detection == 0): 
                    time_catface_detection = time.time()
                if( time.time() - time_catface_detection > 1.5):
                    print("Maximum Confidence for Reference Image: ")
                    print(max(predictions_list))
                    # Thresholds 70%
                    if(max(predictions_list) > 0.02):
                        send_udp_message(b'OK')
                    else: 
                        send_udp_message(b'DENNIED')
                    time.sleep(2.5);
                    time_catface_detection = 0;
                    predictions_list = [] 
                    send_udp_message(b'RUNNING')
                else: 
                    predictions_list.append(prediction[0])  
                print(prediction)
                print("Execution time : [total]", int((end - start)*1000),"ms")
                print("Execution time : [crop]", int((ts_crop - start)*1000),"ms")
                print("Execution time : [resize]", int((ts_resize - ts_crop)*1000),"ms")
                print("Execution time : [cvt]", int((ts_cvt - ts_resize)*1000),"ms")
                print("Execution time : [conv]", int((ts_conv - ts_cvt)*1000),"ms")
                print("Execution time : [prediction]", int((end - ts_conv)*1000),"ms")
                cv2.namedWindow("Isolated Catface")
                cv2.imshow("Isolated Catface", pad_image)
                
    cv2.namedWindow("Camera Image")
    cv2.imshow("Camera Image", frame)
    key = cv2.waitKey(10)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        print("[INFO] Shuting down..")
        break

cv2.destroyAllWindows()
vs.stop()
