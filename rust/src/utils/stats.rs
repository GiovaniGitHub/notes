use std::ops::Mul;
use smartcore::linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix};

pub fn mean(values: &Vec<f32>) -> f32 {
    if values.len() == 0 {
        return 0f32;
    }

    return values.iter().sum::<f32>() / (values.len() as f32);
}


pub fn variance(values: &Vec<f32>) -> f32 {
    if values.len() == 0 {
        return 0f32;
    }

    let mean = mean(values);
    return values
        .iter()
        .map(|x| f32::powf(x - mean, 2 as f32))
        .sum::<f32>()
        / values.len() as f32;
}


pub fn covariance(x_values: &Vec<f32>, y_values: &Vec<f32>) -> f32 {
    if x_values.len() != y_values.len() {
        panic!("x_values and y_values must be of equal length.");
    }

    let length: usize = x_values.len();

    if length == 0usize {
        return 0f32;
    }

    let mut covariance: f32 = 0f32;
    let mean_x = mean(x_values);
    let mean_y = mean(y_values);

    for i in 0..length {
        covariance += (x_values[i] - mean_x) * (y_values[i] - mean_y)
    }

    return covariance / length as f32;
}


pub fn mse(y: DenseMatrix<f32>, y_hat: DenseMatrix<f32>) -> f32{
    let n_rows = y.shape().0 as f32;
    let aux: f32 = y.sub(&y_hat).to_row_vector().iter().map(|x| x.powf(2.0)).sum();
    return aux/n_rows;
}


pub fn mae(y: DenseMatrix<f32>, y_hat: DenseMatrix<f32>) -> f32{
    let n_rows = y.shape().0 as f32;
    let aux: f32 = y.sub(&y_hat).to_row_vector().iter().map(|x| x.abs()).sum();
    return aux/n_rows;
}


pub fn update_weights_mse(x: &DenseMatrix<f32>, y: &DenseMatrix<f32>, y_hat: &DenseMatrix<f32>, lr: f32) -> (DenseMatrix<f32>, f32){
    let (n_rows, n_cols )= x.shape();
    let dif = y.sub(y_hat);
    let mut dw: Vec<f32> = Vec::new();
    for i in 0..n_cols{
        dw.push((1.0/(2.0*(n_rows as f32)))*x.slice(0..n_rows,i..i+1).dot(&dif)*lr)
    }

    let sum_dif: f32 = dif.iter().sum();
    let db: f32 = sum_dif.mul(1.0/(2.0*(n_rows as f32)))*lr;
    return (DenseMatrix::from_array(n_cols,1,&dw), db);
}


pub fn update_weights_mae(x: &DenseMatrix<f32>, y: &DenseMatrix<f32>, y_hat: &DenseMatrix<f32>, lr: f32) -> (DenseMatrix<f32>, f32){
    let (n_rows, n_cols )= x.shape();
    let dif = y.sub(y_hat);
    let dif_abs_sum: f32 = dif.clone().to_row_vector().iter().map(|x| x.abs()).sum();
    let mut dw: Vec<f32> = Vec::new();
    for i in 0..n_cols{
        dw.push((1.0/dif_abs_sum)*x.slice(0..n_rows,i..i+1).dot(&dif)*lr)
    }

    let sum_dif: f32 = dif.iter().sum();
    let db: f32 = sum_dif.mul(-1.0/dif_abs_sum)*lr;
    return (DenseMatrix::from_array(n_cols,1,&dw), db);
}
