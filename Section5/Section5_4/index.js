
const irisData=require("ml-dataset-iris");

const kmeans=require("ml-kmeans")

let values = irisData.getNumbers();
console.log(values)
values = values.map((val) => val.slice(0, 2));
console.log(values)
let result = kmeans(values, 3, {
  seed: 48
});
console.log('kmeans++');
console.log(result.iterations, result.centroids.map((c) => c.error));
console.log(result.centroids);

result = kmeans(values, 3, {
  seed: 8,
  initialization: 'mostDistant'
});
console.log(
  'mostDistant',
  result.iterations,
  result.centroids.map((c) => c.error)
);

result = kmeans(values, 3, {
  seed: 10,
  initialization: 'random'
});
console.log('random');
console.log(result.iterations, result.centroids.map((c) => c.error));
console.log(result.centroids);