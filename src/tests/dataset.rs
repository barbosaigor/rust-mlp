use ndarray::prelude::*;

#[test]
fn read_file() {
    use crate::dataset::read;

    let (input, labels) = read("./train.csv");
    println!("input: {:?}", input.slice(s![390.., 50..60]));
    println!("labels: {:?}", labels.slice(s![0, 100..105]));
    println!("input: {:?}", input.slice(s![.., 0]));
    for (i, v) in input.slice(s![.., 0]).iter().enumerate() {
        if *v != 0.0 {
            print!("[{:?}]: {:?}", i, v);
        }
    }

    for (i, v) in input.iter().enumerate() {
        if *v < 0.0 {
            print!("[{:?}]: {:?}", i, v);
        }
    }
}
