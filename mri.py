# !kill -9 -1
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
import glob

#from google.colab import drive
#drive.mount('/content/drive')

pd_path = '/content/drive/My Drive/Parkinson Dataset/'
control_path = '/content/drive/My Drive/Control Dataset/'

lst = glob.glob(pd_path + '*.npy')
pd = np.zeros((len(lst),80,100,108,1), dtype = 'float32')

for x in range(len(lst)):
  pd[x,:,:,:,0] = np.load(lst[x])

lst = glob.glob(control_path + '*.npy')
control = np.zeros((len(lst),80,100,108,1), dtype = 'float32')

for x in range(len(lst)):
  control[x,:,:,:,0] = np.load(lst[x])

print(pd.shape)
print(control.shape)

x_train = np.concatenate((pd,control),axis = 0)
x_train = x_train/255
y_train = np.zeros((x_train.shape[0], 2), dtype = 'float32')
y_train[0:len(pd), 1] = 1
y_train[len(pd):y_train.shape[0], 0] = 1
print(y_train)
model = Sequential()
model.add(Conv3D(32,(3,3,3), input_shape = (80,100,108,1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv3D(32,(3,3,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size = (2,2,2), strides = 2, padding = 'same'))
model.add(Conv3D(64,(3,3,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv3D(64,(3,3,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size = (4,4,4), strides = 2, padding = 'same'))
model.add(Conv3D(128,(3,3,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv3D(128,(3,3,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size = (4,4,4), strides = 2, padding = 'same'))
model.add(Flatten())
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(2, activation = 'softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr = 0.000005),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=5,
          epochs=100,
          verbose=1,
          validation_data=(x_train, y_train), shuffle = True)

# score = model.evaluate(x_train, y_train, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


