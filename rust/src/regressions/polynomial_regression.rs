use crate::utils::{
    stats::update_weights_mse,
    utils::expand_matrix,
};
use rand::Rng;
use smartcore::linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix};

pub struct PolynomialRegression {
    pub coefficients: DenseMatrix<f32>,
    pub bias: f32,
    pub degree: usize,
}

impl PolynomialRegression {
    pub fn new(degree: usize) -> PolynomialRegression {
        let mut rng = rand::thread_rng();
        let mut coefficients: Vec<f32> = Vec::new();
        for _ in 0..degree {
            coefficients.push(rng.gen::<f32>());
        }

        PolynomialRegression {
            coefficients: DenseMatrix::from_array(degree, 1, &coefficients),
            bias: 1.0,
            degree,
        }
    }

    pub fn predict(&self, x: &DenseMatrix<f32>) -> DenseMatrix<f32> {
        let expanded_matrix = expand_matrix(&x, self.degree);
        let mut y_hat = expanded_matrix.matmul(&self.coefficients);

        y_hat.add_mut(&DenseMatrix::from_vec(
            x.shape().0,
            1,
            &vec![self.bias; expanded_matrix.shape().0],
        ));

        return y_hat;
    }

    pub fn fit(&mut self, x: &DenseMatrix<f32>, y: &DenseMatrix<f32>, epochs: usize, lr: f32) {
        let expanded_matrix = expand_matrix(x, self.degree);
        for _ in 0..epochs {
            let y_hat = self.predict(&x);
            let (dw, db) = update_weights_mse(&expanded_matrix, y, &y_hat, lr);
            self.coefficients.add_mut(&dw);
            self.bias += db;
        }
    }
}
