import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt

#Imports datasests
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), ( test_images, test_labels ) = fashion_mnist.load_data()


#Defining targets
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Proting targets
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()


#Create neural network
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),   #Image size
  keras.layers.Dense(128, activation='relu'),   #Number of inputs neurons and your activation method
  keras.layers.Dense(10, activation='softmax')  #Number of outputs neurons and yout activation method
])

#Configure optimizers
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

#Traning model
model.fit( train_images, train_labels, epochs=100)

#Test model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(
  '\nTeste acurracy: ', test_acc
)
predictions = model.predict(test_images)
print( 
  predictions[2]
)