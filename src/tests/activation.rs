use crate::activation;
use log::info;
use ndarray::prelude::array;

#[test]
fn derivative_re_lu() {
    let input = array![[1., -2., 3.], [-4., 5., 0.]];
    let output = activation::derivative_relu(&input);
    assert_eq!(array![[1., 0., 1.], [0., 1., 0.]], output);
}

#[test]
fn soft_max() {
    let i = array![
        [-68.16620388493381],
        [-329.8711058654097],
        [-265.06210676837225],
        [-128.29778932955807],
        [-55.40857518839206],
        [-9.501913336939275],
        [-83.2217432415665],
        [-1204.7090494269573],
        [-847.5017147392516],
        [2962.05667692332] // <-- this value overflows
    ];
    let output = activation::soft_max(&i);
    println!("{output:?}");
}
