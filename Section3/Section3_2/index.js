const {Matrix} = require('ml-matrix');


export default class LogisticRegressionTwoClasses {
    constructor(options = {}) {
        this.numSteps = options.numSteps || 500000;
        this.learningRate = options.learningRate || 5e-4;
        this.weights = options.weights ? Matrix.checkMatrix(options.weights) : null;
    }

    train(features, target) {
        var weights = Matrix.zeros(1, features.columns);

        for (var step = 0; step < this.numSteps; step++) {
            var scores = features.mmul(weights.transposeView());
            var predictions = sigmoid(scores);

            // Update weights with gradient
            var outputErrorSignal = Matrix.columnVector(predictions).neg().add(target);
            var gradient = features.transposeView().mmul(outputErrorSignal);
            weights = weights.add(gradient.mul(this.learningRate).transposeView());
        }

        this.weights = weights;
    }

    testScores(features) {
        var finalData = features.mmul(this.weights.transposeView());
        var predictions = sigmoid(finalData);
        predictions = Matrix.columnVector(predictions);
        return predictions.to1DArray();
    }

    predict(features) {
        var finalData = features.mmul(this.weights.transposeView());
        var predictions = sigmoid(finalData);
        predictions = Matrix.columnVector(predictions).round();
        return predictions.to1DArray();
    }

    static load(model) {
        return new LogisticRegressionTwoClasses(model);
    }

    toJSON() {
        return {
            numSteps: this.numSteps,
            learningRate: this.learningRate,
            weights: this.weights
        };
    }
}

function sigmoid(scores) {
    scores = scores.to1DArray();
    var result = [];
    for (var i = 0; i < scores.length; i++) {
        result.push(1 / (1 + Math.exp(-scores[i])));
    }
    return result;
}

var X=new Matrix([[1,0,0],[0,0,0],[0,0,0],[0,1,0],[0,0,1],[0,1,0],[1,1,1],[0,0,1],[1,1,0],[0,0,0]]);
var Y=Matrix.columnVector([1,0,0,1,0,0,1,0,1,0]);

// the test set (Xtest, Ytest)
var Xtest=new Matrix([[1,0,0],[0,0,0],[0,0,0],[0,1,0],[0,0,1],[0,1,0],[1,1,1],[0,0,1],[1,1,0],[0,0,0]]);
var Ytest=Matrix.columnVector([1,0,0,1,0,0,1,0,1,0]);


// we will train our model
let logreg = new LogisticRegressionTwoClasses();
logreg.train(X,Y);

// we try to predict the test set
var finalResults = logreg.predict(Xtest);
// Now, you can compare finalResults with the Ytest, which is what you wanted to have.

var count=0;
for(var i=0;i<finalResults.length;i++){
	if(finalResults[i]==Ytest[i])
	 {
	 	count=count+1
	 }
};
console.log("Accuracy "+(count*1.0)/finalResults.length*100);

//Confusion matrix
var truePositive=0;
var trueNegative=0;
var falsePositive=0;
var falseNegative=0;

for(var i=0;i<finalResults.length;i++){
    if(finalResults[i]==Ytest[i])
     {
        if(finalResults[i]==1){
            truePositive=truePositive+1;
        }
        else{
            trueNegative=trueNegative+1;
        }
     }
     else{
      if(finalResults[i]==1){
            falsePositive=falsePositive+1;
        }
        else{
            falseNegative=falseNegative+1;
        }  
     }
};

console.log("True Positive "+truePositive);
console.log("True Negative "+trueNegative);
console.log("False Positive "+falsePositive);
console.log("False Negative "+falseNegative);

var precision = truePositive/(truePositive+falsePositive);
console.log("Precision "+precision);
var recall = truePositive/(truePositive+falseNegative);
console.log("Recall "+recall);