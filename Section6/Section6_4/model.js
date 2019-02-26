/**
 * @license
 * Arish ALi
 * Modified from the Tensorflowjs tutorial  
 *
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

// Hyperparameters.
const LEARNING_RATE = .1;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 100;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;
const optimizer = tf.train.sgd(LEARNING_RATE);

const input_neurons = 784 //number of features in data set
const hidden_neurons = 392 //number of hidden layer neurons
const output_neurons = 10 //number of output layer neurons


// Variables that we want to optimize
const weight_input_hidden =
    tf.variable(tf.randomNormal([input_neurons,hidden_neurons], 0, 0.1));

const weight_hidden_output = tf.variable(
    tf.randomNormal([hidden_neurons,output_neurons], 0, 0.1));

const bias_hidden=tf.variable(
  tf.randomNormal([hidden_neurons], 0, 0.1));

const bias_output=tf.variable(
  tf.randomNormal([output_neurons], 0, 0.1));

// Loss function
function loss(labels, ys) {
  return tf.losses.softmaxCrossEntropy(labels, ys).mean();
}

// Our actual model
function model(inputXs) {
  const xs = inputXs.as2D(-1, IMAGE_SIZE*IMAGE_SIZE);

  // Layer 1
  const layer1 = tf.tidy(() => {
    return xs.as2D(-1, weight_input_hidden.shape[0])
    .matMul(weight_input_hidden)
    .add(bias_hidden);
  });

  // Final layer
  return layer1.as2D(-1, weight_hidden_output.shape[0])
      .matMul(weight_hidden_output)
      .add(bias_output);
}

// Train the model.
export async function train(data, log) {
  const returnCost = true;

  for (let i = 0; i < TRAIN_STEPS; i++) {
    const cost = optimizer.minimize(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE);
      return loss(batch.labels, model(batch.xs));
    }, returnCost);

    log(cost.dataSync(), i);
    await tf.nextFrame();
  }
}

// Predict the digit number from a batch of input images.
export function predict(x) {
  const pred = tf.tidy(() => {
    const axis = 1;
    return model(x).argMax(axis);
  });
  return Array.from(pred.dataSync());
}

// Given a logits or label vector, return the class indices.
export function classesFromLabel(y) {
  const axis = 1;
  const pred = y.argMax(axis);

  return Array.from(pred.dataSync());
}
