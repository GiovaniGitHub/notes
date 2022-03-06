use std::vec;

use smartcore::linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix};
use rand::Rng;
use crate::utils::{
    types::TypeRegression
};
pub struct RBFRegression {
    pub num_center: usize,
    pub centers: DenseMatrix<f32>,
    pub beta: f32,
    pub weight: DenseMatrix<f32>,
    pub type_regression: TypeRegression

}

impl RBFRegression {
    pub fn new(beta: f32, num_center: usize, num_cols: usize,  type_regression: TypeRegression) -> RBFRegression{
        let mut rng = rand::thread_rng();
        let mut coefficients_centers: Vec<f32> = Vec::new();
        let mut coefficients_weight: Vec<f32> = Vec::new();
        for _ in 0..num_center {
            coefficients_weight.push(rng.gen::<f32>());
            for _ in 0..num_cols{
                coefficients_centers.push(rng.gen::<f32>());
            }
        }

        RBFRegression {
            num_center,
            centers: DenseMatrix::from_array(num_center, num_cols, &coefficients_centers),
            beta,
            weight:DenseMatrix::from_array(num_center, 1, &coefficients_weight),
            type_regression,
        }
    }
}