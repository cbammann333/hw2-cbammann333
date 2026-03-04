import tensorflow as tf
import numpy as np
import os
import cv2

model = tf.keras.models.load_model("model.h5")

data = []
labels = []

for img in os.listdir("dataset/positive"):
    img_path = os.path.join("dataset/positive", img)
    image = cv2.imread(img_path)
    image = cv2.resize(image,(32,32))
    data.append(image)
    labels.append(1)

for img in os.listdir("dataset/negative"):
    img_path = os.path.join("dataset/negative", img)
    image = cv2.imread(img_path)
    image = cv2.resize(image,(32,32))
    data.append(image)
    labels.append(0)

X = np.array(data)/255.0
y = np.array(labels)

loss, acc = model.evaluate(X,y)

print("Accuracy:", acc)
