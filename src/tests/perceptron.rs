use crate::activation;
use crate::perceptron::{self, MLP};
use ndarray::prelude::*;
use ndarray::Array;
use rand::Rng;

#[test]
fn mlp() {
    const INPUT_LEN: usize = 5;
    const BATCH_SIZE: usize = 1;

    let mut input = Array::<f64, _>::ones((INPUT_LEN, BATCH_SIZE).f());
    for i in input.iter_mut() {
        *i = rand::thread_rng().gen_range(-1.0..1.0);
    }

    let layer1 = perceptron::PerceptronLayer::new(10, INPUT_LEN, activation::relu);
    let layer2 = perceptron::PerceptronLayer::new(10, 10, activation::relu);

    let mlp = MLP::new();
    let mlp = mlp.push(layer1).push(layer2);

    let output = mlp.synapse(&input);
    println!("{:?}", output);
}

#[test]
fn perceptron_layer() {
    const INPUT_LEN: usize = 5;
    const BATCH_SIZE: usize = 1;
    let p = perceptron::PerceptronLayer::new(10, INPUT_LEN, activation::relu);

    let mut input = Array::<f64, _>::ones((INPUT_LEN, BATCH_SIZE).f());
    for i in input.iter_mut() {
        *i = rand::thread_rng().gen_range(-1.0..1.0);
    }
    let output = p.synapse(&input);
    println!("{:?}", output);
}
