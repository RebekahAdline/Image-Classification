import numpy as np
import pandas as pd
import PIL.Image
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x_train = []
y_train = []
x_test = []
y_test = []

train_loc = 'data/train'
test_loc = 'data/test'

for filename_train in os.listdir(train_loc):
    if filename_train.split('.')[1] == 'jpg':
        img_train = cv2.imread(os.path.join(train_loc,filename_train))
        img_ary_train= PIL.Image.fromarray(img_train,'RGB')
        img_resize_train=img_ary_train.resize((64,64))
        img_train_arr = np.array(img_resize_train)
        x_train.append(img_train_arr)
        y_train.append(filename_train.split('_')[0])

x_train = np.array(x_train).astype('float32')
x_train /= 255

for filename_test in os.listdir(test_loc):
    if filename_test.split('.')[1] == 'jpg':
        img_test = cv2.imread(os.path.join(test_loc,filename_test))
        img_ary_test = PIL.Image.fromarray(img_test,'RGB')
        img_resize_test =img_ary_test.resize((64,64))
        img_test_arr = np.array(img_resize_test)
        x_test.append(img_test_arr)
        y_test.append(filename_test.split('_')[0])

x_test = np.array(x_test).astype('float32')
x_test /= 255

unique_train_classes = len(np.unique(y_train))
unique_test_classes = len(np.unique(y_test))

le = LabelEncoder()
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_train_encode = le.fit_transform(y_train[0])
y_test_encode = le.fit_transform(y_test[0])

# one-hot encode the single digit labels of y_train and y_test and to convert to 4 dimensional array
y_train = tf.keras.utils.to_categorical(y_train_encode, 4)
y_test = tf.keras.utils.to_categorical(y_test_encode, 4)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state = 42)

model = tf.keras.Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(16, activation="relu"))
model.add(Dense(4, activation="softmax"))

model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)
model.summary()

history = model.fit(x_train, y_train,
          batch_size=6, epochs=20, verbose=1,
          validation_data=(x_valid, y_valid), shuffle=True)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

figure = plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.legend(['Training Accuracy','Val Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.legend(['Training Loss','Val Loss'])
plt.show()