const SLR = require('ml-regression').SLR;
let inputs = [80, 60, 10, 20, 30];
let outputs = [20, 40, 30, 50, 60];
 
let regression = new SLR(inputs, outputs);

console.log(regression.predict(80)); 
