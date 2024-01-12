use log::info;
use ndarray::prelude::*;
use ndarray::Array;

// read reads the dataset and converts it into a workable structure
pub fn read(
    f_name: &str,
) -> (
    ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
) {
    info!("Reading dataset");
    const SAMPLES: usize = 1000;
    let mut input: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
        Array::<f64, _>::zeros((784, SAMPLES).f());
    input.fill(-1.0);
    let mut input_labels = Array::<f64, _>::zeros((1, SAMPLES).f());
    let data = std::fs::read(f_name).unwrap();
    let mut reader = csv::Reader::from_reader(data.as_slice());
    for (i, record) in reader.records().enumerate() {
        if i >= SAMPLES {
            break;
        }
        let record = record.unwrap();
        input_labels[[0, i]] = record[0].parse::<f64>().unwrap();
        for j in 1..=784 {
            input[[j - 1, i]] = record[j].parse::<f64>().unwrap() / 255.0;
            assert!(!input[[j - 1, i]].is_nan());
        }
    }
    info!(
        "Success to read dataset, input size: ({:?}, {:?})",
        input.nrows(),
        input.ncols()
    );
    (input, input_labels)
}
