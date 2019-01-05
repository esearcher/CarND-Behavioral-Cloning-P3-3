import csv
import matplotlib.image as mpimg
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

data_dirs = ['../data1', '../data2', '../data3', '../data4', '../data5', '../data6', '../data7', '../data8', '../data9', '../data10', '../data11', '../data12']
lines = []
images = []
measurements = []

for dir in data_dirs:
    with open(dir + '/driving_log.csv') as csvfile:
        new_lines = []
        reader = csv.reader(csvfile)

        for line in reader:
            lines.append([dir + '/IMG/' + line[0].split('/')[-1], line[3]])

sklearn.utils.shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            measurements = []
            for sample in batch_lines:
                img_name = sample[0]
                image = mpimg.imread(img_name)
                images.append(image)
                measurements.append(sample[1])

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), data_format='channels_last'))
model.add(Conv2D(24, (5, 5), activation='relu', strides=(2,2), padding='valid'))
model.add(Conv2D(36, (5, 5), activation='relu', strides=(2,2), padding='valid'))
model.add(Conv2D(48, (5, 5), activation='relu', strides=(2,2), padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='valid'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose = 1)

model.save('model.h5')
