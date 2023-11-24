from utility import TestUtils
import config
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tensorflow.keras.backend as K
def prediction2_custom(model,path1,path2):
    proba = []
    y_preds = []
    ids = []
    notx = 0
    img1 = []
    index = []
    
    generated_frag = TestUtils.fragment_generator(path1,path2)


    y_pred =  model.predict([generated_frag[:,0],generated_frag[:,1]])
    return np.mean(y_pred)

def predictionfrom_csvfile(model,path_images,csv_path):
    data = pd.read_csv(csv_path)
    proba = []
    y_preds = []
    ids = []
    notx = 0
    img1 = []
    index = []
    for i in range(100):
        path1 = path_images + data.iloc[i]["img1_name"]
        path2 = path_images + data.iloc[i]["img2_name"]
        ids.append(data.iloc[i]["img1_name"] + "_" + data.iloc[i]["img2_name"])
        image1 = cv2.imread(path1)
        image2 = cv2.imread(path2)
        generated_frag = list(TestUtils.fragment_generator(path1,path2))
        img1 = img1 + generated_frag
        if len(index) == 0:
            index = index + [[0,len(img1)]]
        else:
            t = index[-1]
            index = index + [[t[-1],t[-1]+len(generated_frag)]]
        print(i)
    img1 = np.array(img1)

    y_pred =  model.predict([img1[:,0],img1[:,1]],batch_size = 1024)
#     print(y_pred.T)
    print(y_pred.shape)
    for i in index:
        if (i[0] == i[1]):
            y_preds.append(0)
            notx += 1
            proba.append(0)
        else:
            if(np.mean(y_pred[i[0]:i[1],0])>.5):
                y_preds.append(1)
            else:
                y_preds.append(0)
            proba.append(np.mean(y_pred[i[0]:i[1],0]))

        print(i)
    return y_preds,proba,ids,notx


model2 = tf.keras.models.load_model("D:/Writer_handwriting_classifier/vgg16_model/archive/vgg16_nn_data.h5")
model2.summary()

# y_preds,proba,ids,notx = predictionfrom_csvfile(model2,"D:/Writer_handwriting_classifier/dataset/dataset/dataset/val/","D:/Writer_handwriting_classifier/dataset/dataset/dataset/val.csv")
print(
    prediction2_custom(model2,"D:/Writer_handwriting_classifier/dataset/dataset/train/P172/A1.jpg","D:/Writer_handwriting_classifier/dataset/dataset/train/P172/B0.jpg"))

