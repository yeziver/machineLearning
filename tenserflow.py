import predictions as predictions
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#loading data
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Cost', 'Sandal'
               , 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255.0
test_images = test_images/255.0

#creating model
model = keras.Sequential([keras.layers.Flatten(input_shape = (28, 28)),
                          keras.layers.Dense(128, activation = "relu"),
                          keras.layers.Dense(10, activation = "softmax")])

#training model
model.compile(optimizer = "adam", los = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 5)

#testing model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)

#using model
predictions = model.predict(test_images)

plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
