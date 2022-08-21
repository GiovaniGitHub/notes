use rand::seq::SliceRandom;
use rand::thread_rng;
use smartcore::linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix};

pub fn expand_matrix(x: &DenseMatrix<f32>, degree: usize) -> DenseMatrix<f32> {
    let mut data_expanded: Vec<f32> = Vec::new();
    for row in 0..x.shape().0 {
        for col in 1..degree {
            data_expanded.push(x.get(row, 0).powf(col as f32));
        }
        data_expanded.push(x.get(row, 0).powf(0.0));
    }
    return DenseMatrix::from_vec(x.shape().0, degree, &data_expanded);
}

pub fn train_test_split(
    x: DenseMatrix<f32>,
    y: DenseMatrix<f32>,
    test_size: f32,
    shuffle: bool,
) -> (
    DenseMatrix<f32>,
    DenseMatrix<f32>,
    DenseMatrix<f32>,
    DenseMatrix<f32>,
) {
    if x.shape().0 != y.shape().0 {
        panic!(
            "x and y should have the same number of samples. |x|: {}, |y|: {}",
            x.shape().0,
            y.shape().0
        );
    }

    if test_size <= 0. || test_size > 1.0 {
        panic!("test_size should be between 0 and 1");
    }

    let n = y.shape().0;

    let n_test = ((n as f32) * test_size) as usize;

    if n_test < 1 {
        panic!("number of sample is too small {}", n);
    }

    let mut indices: Vec<usize> = (0..n).collect();

    if shuffle {
        indices.shuffle(&mut thread_rng());
    }

    let x_train = x.take(&indices[n_test..n], 0);
    let x_test = x.take(&indices[0..n_test], 0);
    let y_train = y.take(&indices[n_test..n], 0);
    let y_test = y.take(&indices[0..n_test], 0);

    (x_train, x_test, y_train, y_test)
}

pub fn accuracy(y_hat: Vec<f32>, y_target: Vec<f32>) -> f32 {
    return (y_hat
        .iter()
        .enumerate()
        .map(|(k, v)| v == &y_target[k])
        .filter(|v| *v)
        .count() as f32)
        / (y_target.len() as f32);
}
