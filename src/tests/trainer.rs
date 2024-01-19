use crate::activation;
use crate::array::one_hot;
use crate::dataset::read;
use crate::perceptron::{self, MLP};
use crate::trainer::Trainer;
use ndarray::prelude::*;
use ndarray::Array;
use rand::Rng;

#[test]
fn train_read() {
    env_logger::init();

    let (input, input_labels) = read("./train.csv");
    let layer1 = perceptron::PerceptronLayer::new(10, input.nrows(), activation::relu);
    let layer2 = perceptron::PerceptronLayer::new(10, layer1.weights.nrows(), activation::soft_max);

    let mlp = MLP::new().push(layer1).push(layer2);

    const ALPHA: f64 = 0.3;
    const EPOCHS: u32 = 150;
    let mut t: Trainer = Trainer::new(mlp, ALPHA, EPOCHS);
    t.train(&input, &input_labels);
}

#[test]
fn train() {
    const INPUT_LEN: usize = 28 * 28;
    const BATCH_SIZE: usize = 1;

    let mut input = Array::<f64, _>::ones((INPUT_LEN, BATCH_SIZE).f());
    for i in input.iter_mut() {
        *i = rand::thread_rng().gen_range(-1.0..1.0);
    }
    let mut input_labels = Array::<f64, _>::ones((1, BATCH_SIZE).f());
    for i in input_labels.iter_mut() {
        *i = rand::thread_rng().gen_range(0..=9) as f64;
    }
    println!("labels: {input_labels:?}");
    let layer1 = perceptron::PerceptronLayer::new(10, INPUT_LEN, activation::relu);
    let layer2 = perceptron::PerceptronLayer::new(10, 10, activation::soft_max);

    let mlp = MLP::new().push(layer1).push(layer2);

    const ALPHA: f64 = 0.01;
    let mut t: Trainer = Trainer::new(mlp, ALPHA, 10);
    t.train(&input, &input_labels);
}

#[test]
fn accuracy() {
    let prediction = array![
        [1.0, 2.0, 3.0, 9.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0]
    ]
    .t()
    .to_owned();

    let labels = array![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]
    .t()
    .to_owned();

    assert_eq!(Trainer::accuracy(&prediction, &labels), 1.0);
}

#[test]
fn accuracy_50() {
    let prediction = array![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    .t()
    .to_owned();

    let labels = array![
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    .t()
    .to_owned();

    assert_eq!(Trainer::accuracy(&prediction, &labels), 0.5);
}

#[test]
fn accuracy_25() {
    let prediction = array![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    .t()
    .to_owned();

    let labels = array![
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    .t()
    .to_owned();

    assert_eq!(Trainer::accuracy(&prediction, &labels), 0.25);
}

#[test]
fn accuracy_real() {
    let prediction = array![
        [0.9939787140657586, 6.439549670232248e-9],
        [2.49212497667679e-6, 0.0025025577848802835],
        [0.00574035410883897, 1.4512860736697141e-5],
        [9.907811738450522e-5, 0.0038255793478814294],
        [0.00013806311197173162, 0.9792530299888811],
        [4.1240404890426184e-5, 0.014360693716527605],
        [8.88675480504997e-12, 1.24637436671909e-6],
        [2.3519657402265474e-8, 7.262059897273201e-11],
        [3.453749248389487e-8, 1.6727740930160397e-7],
        [1.4239765508641112e-13, 4.220613714675754e-5],
    ];

    let labels: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = array![[0.0, 4.0]];
    let labels_onehot = one_hot(&labels, 10);
    println!("labels onehot: {:?}", labels_onehot);

    assert_eq!(Trainer::accuracy(&prediction, &labels_onehot), 1.0);
}

#[test]
fn test_backpropagation_1() {
    let input = array![[1.0], [2.0]];
    let input_labels = array![[1.0]];

    let mut layer1 = perceptron::PerceptronLayer::new(1, input.nrows(), activation::relu);
    layer1.weights = array![[1.0, 1.0]];
    layer1.bias = array![[1.0]];
    let mut layer2 = perceptron::PerceptronLayer::new(2, 1, activation::relu);
    layer2.weights = array![[1.0], [1.0]];
    layer2.bias = array![[1.0], [1.0]];

    let mlp = MLP::new().push(layer1).push(layer2);

    const ALPHA: f64 = 1.0;
    let mut t: Trainer = Trainer::new(mlp, ALPHA, 1);
    t.train(&input, &input_labels);

    assert_eq!(array![[-8.0, -17.0]], t.mlp.layers[0].weights);
    assert_eq!(array![[-8.0]], t.mlp.layers[0].bias);

    assert_eq!(array![[-19.0], [-15.0]], t.mlp.layers[1].weights);
    assert_eq!(array![[-8.0], [-8.0]], t.mlp.layers[1].bias);
}

#[test]
fn test_backpropagation_2() {
    let input = array![[0.5, 0.5], [1.0, 1.0]];
    let input_labels = array![[1.0, 0.0]];

    let mut layer1 = perceptron::PerceptronLayer::new(1, input.nrows(), activation::sigmoid);
    layer1.weights = array![[0.5, 0.5]];
    layer1.bias = array![[0.5]];
    let mut layer2 = perceptron::PerceptronLayer::new(2, 1, activation::soft_max);
    layer2.weights = array![[0.5], [0.5]];
    layer2.bias = array![[0.5], [0.5]];

    let mlp = MLP::new().push(layer1).push(layer2);

    const ALPHA: f64 = 1.0;
    let mut t: Trainer = Trainer::new(mlp, ALPHA, 10);
    t.train(&input, &input_labels);
}
