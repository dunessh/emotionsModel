import numpy as np
import torch
import cv2
import os
from model import LSTMmodel,NeuralModel
import tensorflow as tf
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler




def crop_faces(image_a):
    image = cv2.imread(image_a)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    countImg = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        cv2.imwrite(os.path.join("C:/Users/User/PycharmProjects/FYP/main/cropped_faces/"+os.path.basename(image_a).replace(".jpg", ""), str(countImg) + ".jpg"), roi_color)
        countImg += 1
    # return countImg


def data(img, image_a, count):

    VGG19 = tf.keras.applications.VGG19(weights='imagenet', pooling='max', include_top=False)

    image = mpimg.imread(image_a)
    image = np.expand_dims(image, axis=0).astype('float')/255
    file_path = "Data4/"+os.path.basename(img).replace(".jpg", "")+"/bottleneck_" + str(count)
    features = VGG19.predict(image)
    np.save(file_path, features)
    count += 1
    return count



def predict(path_to):
    file_length = len(os.listdir(path_to))
    count = 0
    valence = 0
    sentiment = 0
    data_file = path_to + "/bottleneck_"
    x = None
    for index in range(file_length):

        file_path = data_file + str(index) + ".npy"

        x_data = np.load(file_path)
        if index != 0:
            x = np.concatenate((x, x_data), axis=0)
        else:
            x = x_data

    scaler = StandardScaler()
    X = scaler.fit_transform(x)
    X = torch.Tensor(X)

    test_size = X.shape[0]

    framework = LSTMmodel(X.shape[1] // 16, 35, 16, 2)
    framework_A = NeuralModel(X.shape[1])


    print("Evalutaion Start ...............")
    batch_size = 1

    framework.load_state_dict(torch.load("./Model_LSTM/model-ckpt-best.txt"))
    framework_A.load_state_dict(torch.load("./Model_Neural/model-ckpt-best.txt"))
    framework_A.eval()
    framework.eval()
    framework.init_hidden(batch_size)

    for param in framework_A.parameters():
        param.requires_grad = False

    for param in framework.parameters():
        param.requires_grad = False


    for i in range(test_size // batch_size):

        framework.init_hidden(batch_size)
        batch_x = X[i * batch_size:(i + 1) * batch_size]
        pred_z = framework_A(batch_x)
        pred_y = framework(batch_x)
        # print(pred_z[:, 0].item())
        # print(pred_y[:, 0].item())
        valence = ((pred_y[:, 0].item())+(pred_z[:, 0].item()))/2
        print(valence)
        sentiment += valence
        count += 1
    print(sentiment/count)
    return sentiment/count







