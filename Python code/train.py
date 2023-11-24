from utility import TrainUtils,CommonUtils
import config
import model
import cv2
from cv2 import imread
import os
import sys
# # import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D,MaxPooling2D,MaxPool2D,Input
from tensorflow.keras.layers import Rescaling,Lambda
import tensorflow.keras.backend as K
from keras.optimizers import Adam,SGD,Adagrad,Adamax
# from keras import layers
# from tensorflow import keras
# import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
# from keras.optimizers import Adam,SGD,Adagrad,Adamax
print("[INFO] loading data...")
img_data, label =TrainUtils.data_loader(config.TRAIN_PATH) #Load images with their label from train folder
print("[INFO] making pairs...")
(img_data, labelTrain) = TrainUtils.make_pairs(img_data[:10000,:,:], label[:10000])#Make pairs of images and their label one with same and one with different label
print("[INFO] splitting data...")
#Split the data into train and test set(validation set for training)
img_data, pairTest,labelTrain,labelTest = train_test_split(img_data, labelTrain, test_size=config.test_size, random_state=0) 
#Learning rate scheduler
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule=TrainUtils.scheduler,verbose=1)

#making model
imgA = Input(shape = config.IMG_SHAPE) 
imgB = Input(shape = config.IMG_SHAPE) 
featureExtractor = model.Convolution_1(config.IMG_SHAPE) 
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
distance = Lambda(TrainUtils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)
model.summary()

#compiling model and setting up optimizer
opt = Adam(learning_rate=config.Learning_rate)
model.compile(optimizer=opt, loss=config.lossfunction,metrics=['accuracy'])


print("[INFO] training model...")
history = model.fit(
    [img_data[:, 0], img_data[:, 1]], labelTrain[:],
    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
    batch_size=config.BATCH_SIZE, 
    epochs=config.EPOCHS,callbacks=[scheduler])