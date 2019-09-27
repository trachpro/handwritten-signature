import numpy as np
import os
from preprocess.normalize import preprocess_signature
import numpy
import cv2

canvas_size = (952, 1360)  # Maximum signature size

def get_dataset(data_path):
    images_name = os.listdir(data_path)
    train_X = []
    train_y = []
    for name in images_name:
        original = cv2.imread(os.path.join(data_path, name), cv2.CV_8UC1)
        processed = preprocess_signature(original, canvas_size)
        newImg = numpy.expand_dims(processed, 2)
        cv2.imshow("test", cv2.dilate(newImg, np.ones((5,5), np.uint8) , iterations=1) )
        cv2.waitKey(0)
        train_X.append(newImg)
        train_y.append(str(int(name[6:8]) - 1))
        print(name[6:8])
    train_X = numpy.asarray(train_X)
    train_y = numpy.asarray(train_y)
    print(train_X.shape)
    return train_X, train_y

def read_new_dataset(path):
    folders = [os.path.join(path,e) for e in os.listdir(path)]
    train_X = []
    train_y = []
    for i, folder in enumerate(folders):
        images = [os.path.join(folder, e) for e in os.listdir(folder)]
        for image_path in images:
            original = cv2.imread(image_path, cv2.CV_8UC1)
            processed = preprocess_signature(original, canvas_size)
            newImg = numpy.expand_dims(processed, 2)
            # newImg = cv2.medianBlur(newImg, 9, 0)
            train_X.append(newImg)
            train_y.append(str(i))
            # cv2.imshow("test", newImg )
            # cv2.waitKey(0)
    train_X = numpy.asarray(train_X)
    train_y = numpy.asarray(train_y)
    print(train_X.shape)
    return train_X, train_y
            
read_new_dataset('/home/tupm/projects/TextRecognitionDataGenerator/out')
cv2.destroyAllWindows()