import tensorflow as tf
import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D,MaxPooling2D,MaxPool2D,Input
from tensorflow.keras.layers import Rescaling,Lambda
import tensorflow.keras.backend as K

def Vgg16(IMG_SHAPE,ration=0.2):
    model = Sequential()
    model.add(Conv2D(input_shape=IMG_SHAPE,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024))
    return model


def deepWriter(input_shape, classes=128):
    # Two different input patches
    patch_1 = Input(shape=input_shape)
    # Convolution_1 shares the same weight
    conv1 = Conv2D(96, kernel_size=5, strides=2, activation='relu')
    out1 = conv1(patch_1)
    # MaxPooling
    MP = MaxPooling2D(3, strides=2)
    out1 = MP(out1)
    # Convolution_2 shares the same weight
    conv2 = Conv2D(256, kernel_size=3, activation='relu')
    out1 = conv2(out1)
    # MaxPooling
    out1 = MP(out1)
    # Convolution_3 shares the same weight
    conv3 = Conv2D(384, kernel_size=3, activation='relu')
    out1 = conv3(out1)
    # Convolution_4 shares the same weight
    conv4 = Conv2D(384, kernel_size=3, activation='relu')
    out1 = conv4(out1)
    # Convolution_5 shares the same weight
    conv5 = Conv2D(256, kernel_size=3, activation='relu')
    out1 = conv5(out1)
    # MaxPooling
    out1 = MP(out1)
    # Flatten
    flat = Flatten()
    out1 = flat(out1)
    # Fully Conneted Layer (FC7)
    FC7 = Dense(1024)
    out1 = FC7(out1)
  # Dropout of 0.5
    # Summation of two outputs
    # Softmax layer
    # Make model and compile
    model = Model(inputs=patch_1, outputs=out1)
 

    return model


def Convolution_1(inputShape,embeddingDim=128):
    # Define the base network (shared weights)
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    #inputs = Rescaling(scale= 1./255 ,offset=0.0)(inputs)   
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
#     x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
#     x = MaxPooling2D(pool_size=2)(x)
#     x = Dropout(0.3)(x)
    
#     x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
#     x = MaxPooling2D(pool_size=2)(x)
#     x = Dropout(0.3)(x)
    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    # build the model
    model = Model(inputs, outputs)
    return model

## can be used as comparator in place of euclidean distance 
# thus neural net learn about similarity between handwritten images
def neural_net_comparator(input_shape, classes=128):
    model_out = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[2048]),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
#     layers.Dense(1024, activation='relu'),
#     layers.Dropout(0.2),
#     layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.1),
#     layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid'),  #last sigmoid for binary
    ])
    return model_out