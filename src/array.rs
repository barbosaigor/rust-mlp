use ndarray;
use ndarray::prelude::*;
use ndarray::Array;
use std::usize;

// onehot creates a matrix with ones in the given position e.g
// [1, 2, 0, 3] -> [[0,1,0,0]]
//                  [0,0,1,0]
//                  [1,0,0,0]
//                  [0,0,0,1]]
pub fn one_hot(
    labels: &Array<f64, Dim<[usize; 2]>>,
    classes: usize,
) -> Array<f64, Dim<[usize; 2]>> {
    let mut onehot = Array::<f64, _>::zeros((classes, labels.len()));
    for (i, arr) in labels.axis_iter(Axis(1)).enumerate() {
        onehot[[arr[0] as usize, i]] = 1.0;
    }
    onehot
}

pub fn max(arr: &Array<f64, Dim<[usize; 1]>>) -> (usize, f64) {
    let mut index = 0;
    let mut r: f64 = *arr.get(0).unwrap();
    for (i, v) in arr.iter().enumerate() {
        if *v > r {
            index = i;
            r = *v;
        }
    }
    (index, r)
}

#[test]
fn test_max() {
    let output = max(&array![1.0, 9.0, 3.0, 10.0, -1.0, 0.5]);
    assert_eq!((3, 10.0), output);
}
