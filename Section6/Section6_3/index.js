import * as tf from '@tensorflow/tfjs';

// We use a sequential model for linear regression
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Select loss and optimizer for model
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Height and weight as the training data 
const height = tf.tensor2d([7, 6.8, 7.2, 6.1, 6.5, 6.7], [6, 1]);
const weight = tf.tensor2d([90, 85, 95, 75, 82, 85], [6, 1]);


// Training the model
model.fit(height, weight, {epochs: 500}).then(() => {
    // Use model to predict weight for height 6ft
    model.predict(tf.tensor2d([6], [1,1])).print();
});