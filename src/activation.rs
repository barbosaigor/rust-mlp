use ndarray::{Array, Dim, ShapeBuilder};

pub type Fun = fn(arr: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>>;

pub fn relu(arr: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
    let mut output = Array::<f64, _>::zeros((arr.nrows(), arr.ncols()).f());
    for i in 0..arr.nrows() {
        for j in 0..arr.ncols() {
            if arr[[i, j]] > 0.0 {
                output[[i, j]] = arr[[i, j]]
            }
        }
    }
    output
}

pub fn derivative_relu(arr: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
    let mut output = Array::<f64, _>::zeros((arr.nrows(), arr.ncols()).f());
    for i in 0..arr.nrows() {
        for j in 0..arr.ncols() {
            if arr[[i, j]] > 0.0 {
                output[[i, j]] = 1.0;
            }
        }
    }
    output
}

pub fn soft_max(arr: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
    let mut output = Array::<f64, _>::zeros((arr.nrows(), arr.ncols()).f());
    for i in 0..arr.nrows() {
        for j in 0..arr.ncols() {
            let v = arr[[i, j]].exp();
            output[[i, j]] = v;
        }
    }
    let s = output.sum();
    output / if s == 0.0 { 1.0 } else { s }
}

pub fn sigmoid(arr: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
    let mut output = Array::<f64, _>::zeros((arr.nrows(), arr.ncols()).f());
    for i in 0..arr.nrows() {
        for j in 0..arr.ncols() {
            output[[i, j]] = sigmoid_scalar(arr[[i, j]]);
        }
    }
    output
}

fn sigmoid_scalar(v: f64) -> f64 {
    if v < -40.0 {
        0.0
    } else if v > 40.0 {
        1.0
    } else {
        1.0 / (1.0 + f64::exp(-v))
    }
}
