use crate::array::{max, one_hot};
use crate::perceptron::MLP;
use log::{debug, info};
use ndarray;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray::ShapeBuilder;
use std::usize;

pub struct Trainer {
    pub mlp: MLP,
    pub alpha: f64,
    pub epochs: u32,
}

impl Trainer {
    pub fn new(mlp: MLP, alpha: f64, epochs: u32) -> Self {
        Self { mlp, alpha, epochs }
    }

    pub fn train(
        &mut self,
        input: &Array<f64, Dim<[usize; 2]>>,
        input_labels: &Array<f64, Dim<[usize; 2]>>,
    ) {
        for epoch in 0..self.epochs {
            let synapses = self.mlp.synapse_breakdown(input);
            let onehot = one_hot(
                input_labels,
                self.mlp.layers[self.mlp.layers.len() - 1].weights.nrows(),
            );
            let bp_params = self.mlp.calculate_bp(&synapses, input, &onehot);
            for (i, (d_w, d_b)) in bp_params.into_iter().enumerate() {
                self.mlp.layers[i].weights =
                    self.mlp.layers[i].weights.clone() - (self.alpha * &d_w.clone());
                self.mlp.layers[i].bias -= self.alpha * d_b;
                debug!(
                    "layer[{:?}].weights: {:?}",
                    i + 1,
                    self.mlp.layers[i].weights
                );
                debug!("layer[{:?}].bias: {:?}", i + 1, self.mlp.layers[i].bias);
            }

            let acc = Self::accuracy(&synapses[synapses.len() - 1].a, &onehot);
            info!("### Epoch {:?}, accuracy: {:?}%", epoch + 1, acc * 100.0);
        }
    }

    pub fn accuracy(
        prediction: &Array<f64, Dim<[usize; 2]>>,
        input_labels: &Array<f64, Dim<[usize; 2]>>,
    ) -> f64 {
        let diff: Array<f64, _> = prediction
            .axis_iter(Axis(1))
            .enumerate()
            .map(|(i, x)| {
                let mut p = Array::<f64, _>::zeros((prediction.nrows()).f());
                let (max_index, _) = max(&x.to_owned());
                p[[max_index]] = 1.0;
                if p.eq(&input_labels.index_axis(Axis(1), i).to_owned()) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        diff.sum() / (input_labels.ncols() as f64)
    }
}
