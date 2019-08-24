import keras
from keras.datasets import mnist
from keras.models import  Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(x_test.shape)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

# 进行标准化 因为每一个pixel都是从0~255
x_train /= 255
x_test /= 255

# 将类别进行一个独热处理
# print(y_train)
n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
# print(y_train[0])

# 设定网络参数
n_hidden_1 = 256
n_classes = 10

training_epochs = 15
batch_size = 100

# CNN
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(n_hidden_1, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs, verbose=1, validation_data=(x_test, y_test))
