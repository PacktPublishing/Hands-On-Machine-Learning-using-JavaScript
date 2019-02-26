const fs = require("fs");
const hog = require("hog-features");
const {default: Image} = require('image-js');
const SVM = require('libsvm-js/asm');
const Kernel = require('ml-kernel');
const range = require('lodash.range');

let options = {
    type: SVM.SVM_TYPES.NU_SVC, 
    kernel : SVM.KERNEL_TYPES.PRECOMPUTED,
    degree : 3,
    nu : 0.1,
    shrinking : false
};

let options_hog = {
    cellSize: 4,
    blockSize: 2,
    blockStride: 1,
    bins: 6,
    norm: "L2"
};

let X_train = [];
let Y_train = [];
let X_test = [];
let Y_test = [];
let K_train;
let K_test;

let kernel;

async function loadData(){
    // We will load the dataset

    async function loadTrainingSet(){
        var lines = fs.readFileSync('labels_train.csv').toString().split("\n");
        for(var i = 0; i < lines.length; i++){
            var elements = lines[i].split(";");
            if(elements.length < 2)
                continue;
            var file = __dirname + "/data/" + elements[0];
            // in the variable X, we will store the HOG of the pictures
            var image = await Image.load(file);
            image = await image.scale({width:100, height:100});
            var descriptor = hog.extractHOG(image, options_hog);
            X_train.push(descriptor);
            Y_train.push(elements[1]);
        }

        kernel = new Kernel('polynomial', {degree: 3, scale: 1/X_train.length});
        K_train = kernel.compute(X_train).addColumn(0, range(1, X_train.length + 1));
    }

    async function loadTestSet(){
        var lines = fs.readFileSync('labels_test.csv').toString().split("\n");
        for(var i = 0; i < lines.length; i++){
            var elements = lines[i].split(";");
            if(elements.length < 2)
                continue;
            var file = __dirname + "/data/" + elements[0];
	    // in the variable X, we will store the HOG of the pictures
            var image = await Image.load(file);
            image = await image.scale({width:100, height:100});
            var descriptor = hog.extractHOG(image, options_hog);
            X_test.push(descriptor);
            Y_test.push(elements[1]);
        }
        K_test = kernel.compute(X_test, X_train).addColumn(0, range(1, X_test.length + 1));
    }

    await loadTrainingSet();
    await loadTestSet();
}

loadData().then(function(){
    // Now, the dataset should be loaded. We will apply the classification
    // Begin of the classification

    var classifier = new SVM(options);

    classifier.train(K_train, Y_train);
    test();

    function test() {
        const result = classifier.predict(K_test);
        const testSetLength = X_test.length;
        const predictionError = error(result, Y_test);
        const accuracy = ((parseFloat(testSetLength)-parseFloat(predictionError))/parseFloat(testSetLength))*100;
	console.log(`Test Set Size = ${testSetLength} and accuracy ${accuracy}%`);
    }

    function error(predicted, expected) {
        let misclassifications = 0;
        for (var index = 0; index < predicted.length; index++) {
            console.log(`${index} => expected : ${expected[index]} and predicted : ${predicted[index]}`);
            if (predicted[index] != expected[index]) {
	        misclassifications++;
   	    }
	}
        return misclassifications;
    }
    fs.writeFileSync('serialized.txt', classifier.serializeModel()); // change this line if you use sth else than SVM

});
