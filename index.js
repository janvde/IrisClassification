const csvFilePath = './iris.csv';
const csv = require('csvtojson');
const KNN = require('ml-knn');

const names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'type']; // For header

const testRatio = 0.8;

//read data
csv({noheader: true, headers: names})
    .fromFile(csvFilePath)
    .then(function (iris) {
        //shuffle array, since the dataset is ordered.
        iris = shuffleArray(iris);
        //normalize the data by converting strings to numbers
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
        console.log(`Training with data set size = ${trainingSetData.length}`);

        //test the model with the test set
        const result = knn.predict(testSetData);
        const predictionError = error(result, testSetPrediction);
        console.log(`Testing with test set size = ${testSetData.length} and number of errors = ${predictionError}`);
    });


/**
 * calculate error as amount of errors between expected and predicted
 * @param predicted array
 * @param expected array
 * @return number of errors
 */
function error(predicted, expected) {
    let errors = 0;
    predicted.forEach(function (prediction, index) {
        if (prediction !== expected[index]) {
            errors++;
        }
    });

    return errors;
}

/**
 * get data, without label
 * @param 2-dimensional array of numbers
 */
function getData(array) {
    return array.map(function (item) {
        return Object.values(item).slice(0, 4);
    })
}

/**
 * get label without data
 *
 * */
function getPrediction(array) {
    return array.map(function (item) {
        return item.type
    })

}

/**
 * convert string numbers to floats and convert flower types to numbers 1-3
 * @param array
 * @return array of normalized data
 */
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

/**
 * randomize array order
 */
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1));
        let temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}