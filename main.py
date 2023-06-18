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

def fetch_data():
    ###########################################################################
    # Load raw data.
    print("Loading data into training/testing raw subsets... ", end="")
    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.mnist.load_data()
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

    return ((x_train, y_train), (x_test, y_test))

((x_train, y_train), (x_test, y_test)) = fetch_data()


###############################################################################
# Build model
def my_custom_accuracy(y_true, y_pred):
    return K.mean(K.equal(tf.cast(x=tf.squeeze(y_true), dtype="int64"),
                          K.argmax(y_pred)))


def build_model(input_shape, int_model_type = 0):
    if int_model_type == 0 :
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=(3,3),
                strides=(1,1),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2,2),
                strides=None, # aka (2,2), the same as pool_size
                padding='valid',
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=64, activation="relu",),
            tf.keras.layers.Dense(units=64, activation="relu",),
            tf.keras.layers.Dense(units=10, activation="softmax",),
        ])
    elif int_model_type == 1 :
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
    else :
        model = None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # "adam"
        #optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #loss="sparse_categorical_crossentropy", # for integer categories
        metrics= [
            my_custom_accuracy,
            tf.keras.metrics.SparseCategoricalAccuracy(), # "accuracy" integers
            # tf.keras.metrics.BinaryAccuracy(threshold=0.5), # 'accuracy' binary integers
            # tf.keras.metrics.CategoricalAccuracy(), # 'accuracy'; one-hot
            # tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3); # "top_k_categorical_accuracy"; one-hot
            ],
        run_eagerly=True,
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
print()


###############################################################################
# Train model
print("Training model:")
history = model.fit(x=x_train, y=y_train, epochs=5, batch_size=32, verbose=2)
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
        y="my_custom_accuracy",
        title="Custom accuracy versus Epochs",
        legend=False)
    acc_plot.set(xlabel="Epochs", ylabel="Custom accuracy")
    plt.savefig("model_training_custom_by_epoch.pdf",
                format="pdf", bbox_inches="tight")


print("Generating plots with model training histories...", end="")
plot_model_training_history(history)
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
        axes[i, 0].text(11., -1.6, 'Label {:d}'.format(y_test_rnd_sample_act))
        axes[i, 1].bar(np.arange(
            len(y_test_rnd_sample_pred)),
            y_test_rnd_sample_pred)
        axes[i, 1].set_xticks(np.arange(len(y_test_rnd_sample_pred)))
        axes[i, 1].set_title(
            "Predicted distribution. Most likely value: {:d}.".format(
                np.argmax(y_test_rnd_sample_pred)))
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


if False :
    print("Sample metrics:")
    # binary classification sigmoid accuracy (binary integer)
    # BinaryAccuracy or SparseCategoricalAccuracy (generalized case)
    y_true = tf.constant([0.0, 1.0, 1.0])
    y_pred = tf.constant([0.4, 0.8, 0.3])
    accuracy = K.mean(K.equal(y_true, K.round(y_pred)))
    print(accuracy.numpy())

    # binary classification softmax accuracy (binary one-hot)
    # CategoricalAccuracy (special binary case)
    y_true = tf.constant([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0],  [0.0,  1.0]])
    y_pred = tf.constant([[0.4, 0.6], [0.3, 0.7], [0.05, 0.95],[0.33, 0.67]])
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
