from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pylab as plt

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" 

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

import numpy as np
import PIL.Image as Image


imagenet_labels = np.array(open('./sample_dataset/labels.txt').read().splitlines())

# Edit file paths here
train_path = ''
test_path = ''
export_path = ''
model_analytics_path = ''

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

training_set = image_generator.flow_from_directory(train_path, target_size=IMAGE_SHAPE)
test_set = test_datagen.flow_from_directory(test_path, target_size=IMAGE_SHAPE)


feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" 
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

feature_extractor_layer.trainable = False


model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(training_set.num_classes)
])

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(training_set.samples/training_set.batch_size)

batch_stats_callback = CollectBatchStats()

r = model.fit_generator(
  training_set, 
  validation_data=test_set,
  epochs=17,
  steps_per_epoch=steps_per_epoch,
  validation_steps=len(test_set),
  callbacks = [batch_stats_callback]
)

print(r.history.keys())
class_names = sorted(training_set.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
class_names

plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(r.history['acc'])
plt.plot(r.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss

plt.subplot(212)
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.savefig(model_analytics_path)

import time
t = time.time()

model.save(export_path, save_format='tf')

export_path

reloaded = tf.keras.models.load_model(export_path)
