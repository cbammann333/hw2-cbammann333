import tensorflow as tf
import numpy as np
import os
import cv2

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

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X,y,epochs=10)

model.save("model.h5")
