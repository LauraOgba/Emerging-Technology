import gzip
import numpy as np

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content = f.read()

    type(file_content)
file_content[0:4]
int.from_bytes(file_content[0:4], byteorder='big')
int.from_bytes(file_content[4:8], byteorder='big')
int.from_bytes(file_content[8:12], byteorder='big')
int.from_bytes(file_content[12:16], byteorder='big')
int.from_bytes(file_content[278:279], byteorder='big')
l = file_content[16:800]
type(l)

image = ~np.array(list(file_content[16:800])).reshape(28,28).astype(np.uint8)

%matplotlib inline
import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')# Adapted from: https://docs.python.org/3/library/gzip.html

import gzip

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    labels = f.read()int.from_bytes(labels[8:9], byteorder="big")
# Import keras.
import keras as kr

# Start a neural network, building it by layers.
model = kr.models.Sequential()

# Add a hidden layer with 1000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=1000, activation='relu', input_dim=784))
# Add a three neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read() 

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()
    
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)inputs = train_img.reshape(60000, 784)
# For encoding categorical variables.
import sklearn.preprocessing as pre

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

print(train_lbl[0], outputs[0])for i in range(10):
    print(i, encoder.transform([i]))model.fit(inputs, outputs, epochs=1, batch_size=100)
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()
    
test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)(encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()


model.predict(test_img[11:12])
