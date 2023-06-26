#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 James James Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
@author: James J Johnson
@url: https://www.linkedin.com/in/james-james-johnson/
"""


###############################################################################
# Load packages

import os
# import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # drop NUMA warnings from TF
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


###############################################################################
# An attempt to use TF 2.12.3 did not work (error about libdevice.10.bc).
# See this link for details:
# https://github.com/tensorflow/tensorflow/issues/58681
# The simplest solution was to downgrade to TF 2.9.1, which works.
print("Tensorflow version: {version}".format(version=tf.__version__))


###############################################################################
# Report CPU/GPU availability.
print()
print("Fitting will be using {int_cpu_count:d} CPU(s).".format(
    int_cpu_count = len(tf.config.list_physical_devices('CPU'))))
print("Fitting will be using {int_gpu_count:d} GPU(s).".format(
    int_gpu_count = len(tf.config.list_physical_devices('GPU'))))
print()

def fetch_data(bool_digits_true_fashion_false = True) :
    ###########################################################################
    # Load raw data.
    print("Loading data into training/testing raw subsets... ", end="")
    if bool_digits_true_fashion_false :
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.mnist.load_data()
        lst_str_classes_labels = [str(i) for i in range(10)]
    else :
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.fashion_mnist.load_data()
        lst_str_classes_labels = [
            'T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle boot',
            ]
    print("done.")
    assert(len(y_train.shape) == 1)
    assert(y_train.shape[0] == x_train.shape[0])
    assert(len(y_test.shape) == 1)
    assert(y_test.shape[0] == x_test.shape[0])
    print("x_train shape: {}.".format(str(x_train.shape)))
    print("y_train shape: {}.".format(str(y_train.shape)))
    print("x_test shape: {}.".format(str(x_test.shape)))
    print("y_test shape: {}.".format(str(y_test.shape)))
    print()

    ###########################################################################
    # Transform data by scaling.
    print("Transforming data...", end="")
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    assert(np.all(x_train <= 1.0) and np.all(x_train >= 0.0))
    assert(np.all(x_test <= 1.0) and np.all(x_test >= 0.0))
    print("done.")
    print("Dependent variable classes in training set: ", ", ".join(
        ["{:.0f}".format(val) for val in np.unique(y_train)]))
    print("Dependent variable classes in testing set: ", ", ".join(
        ["{:.0f}".format(val) for val in np.unique(y_test)]))
    print()

    ###########################################################################
    # Adding the channel dimension with value 1.
    print("Adding channel dimension...", end="")
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    print("done.")
    print("x_train shape: {}.".format(str(x_train.shape)))
    print("y_train shape: {}.".format(str(y_train.shape)))
    print("x_test shape: {}.".format(str(x_test.shape)))
    print("y_test shape: {}.".format(str(y_test.shape)))
    print()

    return (lst_str_classes_labels, (x_train, y_train), (x_test, y_test))

(lst_str_classes_labels, (x_train, y_train), (x_test, y_test)) = fetch_data(
    bool_digits_true_fashion_false = False)


###############################################################################
# Build model
def custom_metric_accuracy(y_true, y_pred):
    return K.mean(K.equal(tf.cast(x=tf.squeeze(y_true), dtype="int64"),
                          K.argmax(y_pred)))

def custom_kernel_initializer_normal(shape, dtype=None) :
    return K.random_normal(shape, dtype=dtype)

def build_model(input_shape, int_model_type = 0):
    if int_model_type == 0 :
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same", # 'valid'
                activation="relu",
                kernel_initializer='random_uniform',
                bias_initializer="zeros",
                input_shape=input_shape, # (None, 28, 28, 1)
                # "channels_first" or "channels_last" (dft) for "input_shape"
                data_format="channels_last",
            ), # (None, W, H, 32);
            # Conv2D: W = floor((w + 2*p - f) / s + 1)
            #         H = floor((h + 2*p - f) / s + 1)
            # padding="same":
            # Conv2D: W = floor((28 + 2*1 - 3) / 1 + 1)
            #         H = floor((28 + 2*1 - 3) / 1 + 1)
            # padding="valid":
            # Conv2D: W = floor((28 + 2*0 - 3) / 1 + 1)
            #         H = floor((28 + 2*0 - 3) / 1 + 1)
            tf.keras.layers.MaxPooling2D(
                pool_size=(2,2),
                strides=None, # If None, it will default to pool_size.
                padding='valid',
            ), # (None, W, H, 32);
            # MaxPool2D: W = floor((w - f) / s + 1)
            #            H = floor((h - f) / s + 1)
            # MaxPool2D: W = floor((28 - 2) / 2 + 1);
            #            H = floor((28 - 2) / 2 + 1)
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same", # 'valid'
                activation="relu",
                kernel_initializer='random_uniform',
                bias_initializer="zeros",
                input_shape=input_shape, # (None, 28, 28, 1)
                # "channels_first" or "channels_last" (dft) for "input_shape"
                data_format="channels_last",
            ), # (None, W, H, 32);
            tf.keras.layers.MaxPooling2D(
                pool_size=(2,2),
                strides=None, # If None, it will default to pool_size.
                padding='valid',
            ), # (None, W, H, 32);
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=64,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.06), 
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.1), 
                ),
            tf.keras.layers.Dense(
                units=64,
                activation="relu",
                kernel_initializer=tf.keras.initializers.Orthogonal(
                    gain=1.1, seed=None), 
                bias_initializer=tf.keras.initializers.Constant(
                    value=0.3),
                ),
            tf.keras.layers.Dense(
                units=10,
                activation="softmax",
                kernel_initializer=custom_kernel_initializer_normal,
                ),
            # The above layer can be split into three layers:
            #
            # tf.keras.layers.Dense(units=10, activation='linear'),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Softmax(),
            #
            #Dense(units=1,activation="linear"), # See "from_logits=True" in model.compile!
        ])
    elif int_model_type == 1 :
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(units=64, activation='tanh'),
            tf.keras.layers.Dense(units=64, activation='relu'), # 'elu'
            tf.keras.layers.Dense(units=10, activation='softmax'),
            # Use activation='softmax' for units = 1 in the last layer
        ])
    elif int_model_type == 2 :
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])
    else :
        model = None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # "adam"
        #optimizer='sgd', # "adam", "rmsprop", "adadelta"
        #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,
        #                                  nesterov=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #loss="sparse_categorical_crossentropy", # y_train: (num_samples, ) # sparse representation vector (integer enumerations)
        #loss='binary_crossentropy',
        #loss="mean_squared_error",
        #loss="categorical_crossentropy", # y_train: (num_samples, num_classes) # one-hot vector (0's or 1's)
        #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # See activation="linear"
        metrics= [
            custom_metric_accuracy,
            tf.keras.metrics.SparseCategoricalAccuracy(), # "accuracy" integers
            # tf.keras.metrics.BinaryAccuracy(threshold=0.5), # 'accuracy' binary integers
            # tf.keras.metrics.CategoricalAccuracy(), # 'accuracy'; one-hot
            # tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3); # "top_k_categorical_accuracy"; one-hot
            # tf.keras.metrics.MeanAbsoluteError() # for regressions!
            ],
        #metrics=["accuracy", "mae",],
        # Metrics aree computed for each epoch during training along with
        # evaluation of the loss function on the training data.
        # run_eagerly=True, # makes the run slower!!!
    )
    # When you pass the strings "accuracy" or "acc", we convert this to one of
    # 1. tf.keras.metrics.BinaryAccuracy,
    # 2. tf.keras.metrics.CategoricalAccuracy
    # 3. tf.keras.metrics.SparseCategoricalAccuracy
    # based on the loss function used and the model output shape.
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
    # https://stackoverflow.com/questions/69218874/why-is-metrics-tf-keras-metrics-accuracy-giving-an-error-but-metrics-ac

    return model

print("Building model with input shape {}... ".format(x_train[0].shape), end="")
model = build_model(x_train[0].shape)
print("done.")
#print(model.optimizer)
#print(model.loss)
#print(model.compiled_metrics)
#print(model.compiled_metrics._metrics)
print()


###############################################################################
# Access podel paramaters and plot their distributions as histograms.
def plot_hist_model_params(model) :

    # Exclude layers which do not have weights (e.g. Flatten, MaxPooling2D)
    weight_layers = [layer for layer in model.layers
                     if len(layer.weights) > 0]
    num_weight_layers_to_plot = len(weight_layers)

    fig, axes = plt.subplots(num_weight_layers_to_plot, 2, figsize=(13,17))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, weight_layer in enumerate(weight_layers):
        for j in [0, 1]:
            axes[i, j].hist(
                weight_layer.weights[j].numpy().flatten(),
                align='left')
            axes[i, j].set_title(weight_layer.weights[j].name)
    plt.savefig("model_params_histograms.pdf",
                format="pdf", bbox_inches="tight")

print("Generating histograms with model parameters...", end="")
plot_hist_model_params(model=model)
print("done.")
print()


###############################################################################
# Train model

class CStopModelTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, flt_min_train_accuracy) :
        self._flt_min_train_accuracy = flt_min_train_accuracy
    def on_epoch_end(self, epoch, logs={}) :
        if(logs.get('sparse_categorical_accuracy') is not None and
           logs.get('sparse_categorical_accuracy') >=
           self._flt_min_train_accuracy) :
          self.model.stop_training = True

print("Training model:")
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    batch_size=32,
    verbose=2,
    callbacks=[CStopModelTrainingCallback(
        flt_min_train_accuracy = 0.93)]
    )
print()
print(model.summary())
print()


###############################################################################
# Plot model training history
def plot_model_training_history(history) :

    df_history = pd.DataFrame(history.history)

    acc_plot = df_history.plot(
        y="loss",
        title = "Loss versus Epochs",
        legend=False)
    acc_plot.set(xlabel="Epochs", ylabel="Loss")
    plt.savefig("model_training_loss_by_epoch.pdf",
                format="pdf", bbox_inches="tight")

    acc_plot = df_history.plot(
        y="sparse_categorical_accuracy", # "accuracy"
        title="Accuracy versus Epochs",
        legend=False)
    acc_plot.set(xlabel="Epochs", ylabel="Accuracy")
    plt.savefig("model_training_accuracy_by_epoch.pdf",
                format="pdf", bbox_inches="tight")

    acc_plot = df_history.plot(
        y="custom_metric_accuracy",
        title="Custom accuracy versus Epochs",
        legend=False)
    acc_plot.set(xlabel="Epochs", ylabel="Custom accuracy")
    plt.savefig("model_training_custom_by_epoch.pdf",
                format="pdf", bbox_inches="tight")


print("Generating plots with model training histories...", end="")
plot_model_training_history(history = history)
print("done.")
print()


###############################################################################
# Evaluate model.
print("Evaluating model on testing set...", end="")
loss_test, custom_metric_test, accuracy_test = model.evaluate(x_test, y_test)
print("done.")
print("Test loss: {loss_test:.4f}".format(
    loss_test=loss_test))
print("Test accuracy: {accuracy_test:.4f}".format(
    accuracy_test=accuracy_test))
print("Test custom accuracy metric: {custom_metric_test:.4f}".format(
    custom_metric_test=custom_metric_test))
print()


###############################################################################
# Predict with the model for random samples.
def generate_random_subset(num_samples):
    rand_generator = np.random.default_rng(12345)
    arr_rand_indices = rand_generator.choice(
        a=x_test.shape[0], size=num_samples, replace=False)
    x_test_rnd_subset = x_test[arr_rand_indices, ...]
    y_test_rnd_subset_act = y_test[arr_rand_indices, ...]
    return (x_test_rnd_subset, y_test_rnd_subset_act)

num_samples = 5
print("Predicting for {:d} randomly selected samples.".format(num_samples))
(x_test_rnd_subset, y_test_rnd_subset_act) = \
    generate_random_subset(num_samples = num_samples)
y_test_rnd_subset_pred = model.predict(x_test_rnd_subset)
print()


###############################################################################
# Generating plots with predicted results.
def plot_predicted_distributions(
        x_test_rnd_subset,
        y_test_rnd_subset_act,
        y_test_rnd_subset_pred):
    num_samples = x_test_rnd_subset.shape[0]
    fig, axes = plt.subplots(num_samples, 2, figsize=(18, 14))
    fig.subplots_adjust(hspace=0.5, wspace=-0.15)
    for i, (y_test_rnd_sample_pred,
            x_test_rnd_sample,
            y_test_rnd_sample_act) in enumerate(
            zip(y_test_rnd_subset_pred,
                x_test_rnd_subset,
                y_test_rnd_subset_act)):
        axes[i, 0].imshow(np.squeeze(x_test_rnd_sample))
        axes[i, 0].get_xaxis().set_visible(False)
        axes[i, 0].get_yaxis().set_visible(False)
        axes[i, 0].text(11., -1.6, 'Label #{:d}: {}'.format(
            y_test_rnd_sample_act,
            lst_str_classes_labels[y_test_rnd_sample_act]))
        axes[i, 1].bar(np.arange(
            len(y_test_rnd_sample_pred)),
            y_test_rnd_sample_pred)
        axes[i, 1].set_xticks(np.arange(len(y_test_rnd_sample_pred)))
        int_best_choice_ind = np.argmax(y_test_rnd_sample_pred)
        axes[i, 1].set_title(
            "Predicted distribution. Most likely value: {}.".format(
                lst_str_classes_labels[int_best_choice_ind]))
    plt.savefig(
        "selected_images_predictions.pdf",
        format="pdf",
        bbox_inches="tight")

print("Generating plots with predicted results... ", end="")
plot_predicted_distributions(
    x_test_rnd_subset,
    y_test_rnd_subset_act,
    y_test_rnd_subset_pred)
print("done.")
print()


###############################################################################
# Choose a random sample and predict its class label
print("Selecting random sample... ", end="")
rand_generator = np.random.default_rng(12345)
inx_test_rnd_sample = rand_generator.choice(x_test.shape[0])
# inx_test_rnd_sample = 30
x_test_rnd_sample = x_test[inx_test_rnd_sample]
y_test_rnd_sample_act = y_test[inx_test_rnd_sample]
print("done.")
np.set_printoptions(linewidth=320)
print(f'Pixel array:\n {(np.round((x_test_rnd_sample*255)[:,:,0])).astype(int)}')
print("Selected random sample with index {:d}.".format(inx_test_rnd_sample))
print("Displaying random sample... ", end="")
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
plt.imshow(x_test_rnd_sample)
plt.show()
print("done.")
print("The shape of the random sample: {}.".format(
    str(x_test_rnd_sample.shape)))
x_test_rnd_sample_batch = x_test_rnd_sample[np.newaxis,...]
print("The shape of the random sample batch: {}.".format(
    str(x_test_rnd_sample_batch.shape)))
print("Predicting lable of the random sample:")
y_test_rnd_sample_pred = model.predict(x_test_rnd_sample_batch)
print("Probabilities of predicted labels of the random sample:")
print(np.round(y_test_rnd_sample_pred,4))
int_max_prob_idx = np.argmax(y_test_rnd_sample_pred)
print("The index of the biggest probability: {:d}".format(
    int_max_prob_idx))
print(f"Actual class index: {y_test_rnd_sample_act}")
print(f"Actual class name: {lst_str_classes_labels[y_test_rnd_sample_act]}")
print(f"Predicted class index: {int_max_prob_idx}")
print(f"Predicted class name: {lst_str_classes_labels[int_max_prob_idx]}")
print()


###############################################################################
# Plot images from Conv2D and MaxPool2D layers.
print("Actual class indices (subsample):")
print(y_test[:32])

IMAGE1_IDX = 2
IMAGE2_IDX = 3
IMAGE3_IDX = 5
IMAGE4_IDX = 15
IMAGE5_IDX = 24

CONV_FILTER_IDX = 10 # change these from [0; 31]

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(
    inputs = model.input, outputs = layer_outputs)

_, ax = plt.subplots(5,4)
for x in range(0,4):
  f1 = activation_model.predict(x_test[IMAGE1_IDX].reshape(1, 28, 28, 1),
                                verbose=False)[x]
  ax[0,x].imshow(f1[0, : , :, CONV_FILTER_IDX], cmap='inferno')
  ax[0,x].grid(False)
  
  f2 = activation_model.predict(x_test[IMAGE2_IDX].reshape(1, 28, 28, 1),
                                verbose=False)[x]
  ax[1,x].imshow(f2[0, : , :, CONV_FILTER_IDX], cmap='inferno')
  ax[1,x].grid(False)
  
  f3 = activation_model.predict(x_test[IMAGE3_IDX].reshape(1, 28, 28, 1),
                                verbose=False)[x]
  ax[2,x].imshow(f3[0, : , :, CONV_FILTER_IDX], cmap='inferno')
  ax[2,x].grid(False)

  f4 = activation_model.predict(x_test[IMAGE4_IDX].reshape(1, 28, 28, 1),
                                verbose=False)[x]
  ax[3,x].imshow(f4[0, : , :, CONV_FILTER_IDX], cmap='inferno')
  ax[3,x].grid(False)

  f5 = activation_model.predict(x_test[IMAGE5_IDX].reshape(1, 28, 28, 1),
                                verbose=False)[x]
  ax[4,x].imshow(f5[0, : , :, CONV_FILTER_IDX], cmap='inferno')
  ax[4,x].grid(False)


if True :
    ###########################################################################
    # Stylized examples that show how to compute customized metrics
    # See also:
    # https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05

    print("Extras: compute Accuracy metric with stylized examples:")
    # binary classification sigmoid accuracy (binary integer)
    # BinaryAccuracy or SparseCategoricalAccuracy (generalized case)
    y_true = tf.constant([0.0, 1.0, 1.0])
    y_pred = tf.constant([0.4, 0.8, 0.3])
    accuracy = K.mean(K.equal(y_true, K.round(y_pred)))
    print(accuracy.numpy())

    # binary classification softmax accuracy (binary one-hot)
    # CategoricalAccuracy (special binary case)
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0],])
    y_pred = tf.constant([[0.6, 0.4], [0.2, 0.8], [0.7, 0.3],])
    accuracy = K.mean(K.equal(y_true, K.round(y_pred)))
    print(accuracy.numpy())

    # multiclass classification argmax accuracy (multiclass one-hot)
    # CategoricalAccuracy
    y_true = tf.constant([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]])
    y_pred = tf.constant([
        [0.4, 0.6, 0.0, 0.0],
        [0.3, 0.2, 0.1, 0.4],
        [0.05, 0.35, 0.5, 0.1]])
    accuracy = K.mean(K.equal(
        K.argmax(y_true, axis=-1),
        K.argmax(y_pred, axis=-1)))
    print(accuracy.numpy())

    # multiclass classification argmax accuracy (multiclass integer vs one-hot)
    y_true = tf.constant([
        [1.0],
        [0.0],
        [2.0]])
    y_pred = tf.constant([
        [0.4, 0.6, 0.0, 0.0],
        [0.3, 0.2, 0.1, 0.4],
        [0.05, 0.35, 0.5, 0.1]])
    accuracy = K.mean(K.equal(
        tf.cast(x=tf.squeeze(y_true), dtype="int64"),
        K.argmax(y_pred)))
    print(accuracy.numpy())
