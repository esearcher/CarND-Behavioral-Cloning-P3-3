import csv
import matplotlib.image as mpimg
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

data_dirs = ['../data1', '../data2', '../data3', '../data4', '../data5', '../data6', '../data7', '../data8', '../data9', '../data10']
lines = []
images = []
measurements = []

for dir in data_dirs:
    print(dir)
    with open(dir + '/driving_log.csv') as csvfile:
        new_lines = []
        reader = csv.reader(csvfile)

        for line in reader:
            new_lines.append(line)

        for line in new_lines:
            # Read the image corresponding that line
            image = mpimg.imread(dir + '/IMG/' + line[0].split('/')[-1])
            images.append(image)
            # Read the steering measurement corresponding that line
            measurement = float(line[3])
            measurements.append(measurement)

        lines.append(new_lines)

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), data_format='channels_last'))
model.add(Conv2D(24, (5, 5), activation='relu', strides=(2,2), border_mode='valid'))
model.add(Conv2D(36, (5, 5), activation='relu', strides=(2,2), border_mode='valid'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
model.add(Conv2D(48, (5, 5), activation='relu', strides=(2,2), border_mode='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2,2), border_mode='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), border_mode='valid'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=3, validation_split=0.2, shuffle=True)

model.save('model.h5')
