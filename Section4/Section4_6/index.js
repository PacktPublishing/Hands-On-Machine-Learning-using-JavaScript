const fs = require("fs");
const RFClassifier = require('ml-random-forest').RandomForestClassifier;
const DTClassifier = require('ml-cart').DecisionTreeClassifier;


let X_train = [];
let Y_train = [];
let X_test = [];
let Y_test = [];


var options = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25
};

var dtOptions = {
  gainFunction: 'gini',
  maxDepth: 10,
  minNumSamples: 3
};

async function loadData(){
    // We will load the dataset

    async function loadTrainingSet(){
        var lines = fs.readFileSync('churn_train_csv.csv').toString().split("\n");
        for(var i = 0; i < 1000; i++){
            var elements = lines[i].split(",");
            elements=elements.map(Number);
            Y_train.push(elements.pop());
            X_train.push(elements.map(Number));
        }

}
    async function loadTestSet(){
        var lines = fs.readFileSync('churn_test_csv.csv').toString().split("\n");
        for(var i = 0; i < 333; i++){
            var elements = lines[i].split(",");
            elements=elements.map(Number);
            Y_test.push(elements.pop());
            X_test.push(elements);
        }
}

    await loadTrainingSet();
    await loadTestSet();
}

loadData().then(function(){
    // Now, the dataset should be loaded. We will apply the classification
    // Begin of the classification
    var classifier = new RFClassifier(options);

    classifier.train(X_train,Y_train);
    var result = classifier.predict(X_test);

	var count=0;
	for(var i=0;i<result.length;i++){
		if(result[i]==Y_test[i])
		 {
		 	count=count+1
		 }
	};
	console.log(count/result.length*100);

	var dtClassifier = new DTClassifier(dtOptions);

    dtClassifier.train(X_train,Y_train);
    var dtResult = dtClassifier.predict(X_test);

	var count=0;
	for(var i=0;i<dtResult.length;i++){
		if(dtResult[i]==Y_test[i])
		 {
		 	count=count+1
		 }
	};
	console.log(count/dtResult.length*100);
});
