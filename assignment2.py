import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(fn, label_id, num_class):
    data = pd.read_csv(fn, header=None)
    data, labels = process(data, label_id, num_class)
    return np.array(data), np.array(labels)

def process(data, label_id, num_class):
    labels = data.pop(label_id)
    labels = labels-start
    return data, labels

'''
label_id = 9 
num_class = 7
start = 1
data, labels = load_data('data/UCI/glass/glass.data', label_id, num_class)
'''

label_id = 8 
num_class = 2
start = 0
data, labels = load_data('data/UCI/pima-indians-diabetes/pima-indians-diabetes.data', label_id, num_class)

model = tf.keras.Sequential([
  layers.Dense(64, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(16, activation='relu'),
  layers.Dropout(.5),
  layers.Dense(num_class)
])

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = 'adam', metrics=['accuracy'])

history = model.fit(data, labels, validation_split=0.5, epochs=1000)
def plot_loss(history, label):
  plt.semilogy(history.epoch,  history.history['loss'], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'], label='Val '+label, linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()

plot_loss(history,'')
plt.show()

def plot_acc(history, label):
  plt.semilogy(history.epoch,  history.history['accuracy'], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_accuracy'], label='Val '+label, linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()

plot_acc(history, '')
plt.show()
