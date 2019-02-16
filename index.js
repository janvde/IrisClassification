const csvFilePath = './iris.csv';
const csv = require('csvtojson');
const KNN = require('ml-knn');

const names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'type']; // For header

const testRatio = 0.8;

let iris = [];

//read data
csv({noheader: true, headers: names})
    .fromFile(csvFilePath)
    .then(function (jsonObj) {
        iris = jsonObj
    })
    .finally(function () {
        //data preparation
        iris = shuffleArray(iris);
        iris = normalize(iris);

        //split into training and test set
        let trainingSet = iris.slice(0, testRatio * iris.length);
        let testSet = iris.slice(testRatio * iris.length);

        //split training set in data and prediction set
        let trainingSetData = getData(trainingSet);
        let trainingSetPrediction = getPrediction(trainingSet);

        //slit test set in data and prediction set
        let testSetData = getData(testSet);
        let testSetPrediction = getPrediction(testSet);

        //train the model with the training set
        var knn = new KNN(trainingSetData, trainingSetPrediction);
        console.log(`Training with data set size = ${trainingSetData.length}`)

        //test the model with the test set
        const result = knn.predict(testSetData);
        const predictionError = error(result, testSetPrediction);
        console.log(`Testing with test set size = ${testSetData.length} and number of errors = ${predictionError}`);


    });


function error(predicted, expected) {
    let misclassifications = 0;
    for (var index = 0; index < predicted.length; index++) {
        if (predicted[index] !== expected[index]) {
            misclassifications++;
        }
    }
    return misclassifications;
}

function getData(array) {
    return array.map(function (item) {
        return Object.values(item).slice(0,4);
    })
}

function getPrediction(array) {
    return array.map(function (item) {
        return item.type
    })

}

function normalize(array) {
    return array.map(function (item) {
        let type = 0;
        if (item.type === 'Iris-setosa') type = 0;
        else if (item.type === 'Iris-versicolor') type = 1;
        else type = 2;

        return {
            sepalLength: parseFloat(item.sepalLength),
            sepalWidth: parseFloat(item.sepalWidth),
            petalLength: parseFloat(item.petalLength),
            petalWidth: parseFloat(item.petalWidth),
            type: type
        }
    });
}


function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}