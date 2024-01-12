use crate::activation;
use log::debug;
use ndarray;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray::ShapeBuilder;
use rand::Rng;
use std::usize;

pub struct PerceptronLayer {
    pub weights: Array<f64, Dim<[usize; 2]>>,
    pub bias: Array<f64, Dim<[usize; 2]>>,
    pub activation: activation::Fun,
}

impl PerceptronLayer {
    pub fn new(neurons: usize, conns: usize, activation: activation::Fun) -> Self {
        PerceptronLayer {
            weights: random_array(neurons, conns),
            bias: random_array(neurons, 1),
            activation: activation,
        }
    }

    pub fn synapse(&self, input: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
        (self.activation)(&(self.weights.dot(input) + &self.bias))
    }
}

fn random_array(rows: usize, cols: usize) -> ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> {
    let mut arr = Array::<f64, _>::zeros((rows, cols).f());
    for i in arr.iter_mut() {
        *i = rand::thread_rng().gen_range(-0.5..0.5);
    }
    arr
}

pub struct SynapseBreakdown {
    pub z: Array<f64, Dim<[usize; 2]>>,
    pub a: Array<f64, Dim<[usize; 2]>>,
}

pub struct MLP {
    pub layers: Vec<PerceptronLayer>,
}

impl MLP {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn push(mut self, layer: PerceptronLayer) -> Self {
        self.layers.push(layer);
        self
    }

    // synapse does a straightforward synapse operation throughout the whole network
    pub fn synapse(&self, input: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
        let mut output = input.clone();
        for layer in self.layers.iter() {
            output = layer.synapse(&output);
        }
        output
    }

    // synapse_breakdown breaks down all synapse operations throughout the whole network
    pub fn synapse_breakdown(&self, input: &Array<f64, Dim<[usize; 2]>>) -> Vec<SynapseBreakdown> {
        let z1 = self.layers[0].weights.dot(input) + &self.layers[0].bias;
        let a1: Array<f64, Dim<[usize; 2]>> = (self.layers[0].activation)(&z1);
        assert!(
            Self::is_valid(&z1),
            "layer[0]: z \n\tweights: {:?}\n\tin: {:?}\n\tbias: {:?}",
            self.layers[0].weights,
            input,
            self.layers[0].bias
        );
        assert!(
            Self::is_valid(&a1),
            "layer[0]: activation \n\tweights: {:?}\n\tin: {:?}\n\tbias: {:?}",
            self.layers[0].weights,
            input,
            self.layers[0].bias
        );
        let mut output = Vec::new();
        output.push(SynapseBreakdown { z: z1, a: a1 });
        if self.layers.len() == 1 {
            return output;
        }
        for layer in self.layers[1..].iter() {
            let previous_activation = &output[output.len() - 1].a;
            let z = layer.weights.dot(previous_activation) + &layer.bias;
            let a = (layer.activation)(&z);
            assert!(
                Self::is_valid(&z),
                "layer[1]: z \n\tweights: {:?}\n\tin: {:?}\n\tbias: {:?}",
                self.layers[1].weights,
                previous_activation,
                self.layers[1].bias
            );
            assert!(
                Self::is_valid(&a),
                "layer[1]: activation \n\tweights: {:?}\n\tin: {:?}\n\tbias: {:?}",
                self.layers[1].weights,
                previous_activation,
                self.layers[1].bias
            );
            output.push(SynapseBreakdown { z, a });
        }
        output
    }

    fn is_valid(arr: &Array<f64, Dim<[usize; 2]>>) -> bool {
        !arr.iter().any(|e| e.is_infinite() || e.is_nan())
    }

    // TODO: make it dynamic
    pub fn calculate_bp(
        &self,
        synapses: &Vec<SynapseBreakdown>,
        input: &Array<f64, Dim<[usize; 2]>>,
        onehot: &Array<f64, Dim<[usize; 2]>>,
    ) -> Vec<(Array<f64, Dim<[usize; 2]>>, f64)> {
        let z1 = &synapses[0].z;
        let a1 = &synapses[0].a;
        let a2 = &synapses[1].a;
        let samples = input.ncols();
        debug!("samples: {samples:?}");
        debug!("a2: {a2:?}");
        debug!("a1: {a1:?}");
        debug!("z1: {z1:?}");
        debug!("input: {input:?}");

        let dZ2 = a2 - onehot;
        debug!("dZ2: {dZ2:?}");
        let dW2 = dZ2.dot(&a1.t().to_owned()) * (1.0 / samples as f64);
        let dB2 = (1.0 / samples as f64) * dZ2.sum();

        let dZ1 = self.layers[1].weights.t().to_owned().dot(&dZ2) * activation::derivative_relu(z1);
        debug!("dZ1: {dZ1:?}");
        let dW1 = (1.0 / samples as f64) * dZ1.dot(&input.t().to_owned());
        let dB1 = (1.0 / samples as f64) * dZ1.sum();

        debug!("dW1: {dW1:?}");
        debug!("dB1: {dB1:?}");
        debug!("dW2: {dW2:?}");
        debug!("dB2: {dB2:?}");

        let mut d_params = Vec::<(Array<f64, Dim<[usize; 2]>>, f64)>::new();
        d_params.push((dW1, dB1));
        d_params.push((dW2, dB2));
        d_params
    }
}
