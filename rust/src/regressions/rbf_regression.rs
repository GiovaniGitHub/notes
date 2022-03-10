use smartcore::linalg::{
    naive::dense_matrix::DenseMatrix, qr::QRDecomposableMatrix, svd::SVDDecomposableMatrix,
    BaseMatrix, BaseVector,
};

use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::utils::types::TypeFactoration;
use crate::utils::utils::expand_matrix;

pub struct RBFRegression {
    pub num_center: usize,
    pub centers: DenseMatrix<f32>,
    pub beta: f32,
    pub weight: DenseMatrix<f32>,
    pub type_factoration: Option<TypeFactoration>,
}

impl RBFRegression {
    pub fn new(
        beta: f32,
        num_center: usize,
        num_cols: usize,
        type_factoration: Option<TypeFactoration>,
    ) -> RBFRegression {
        let mut coefficients_centers: Vec<f32> = Vec::new();
        let mut coefficients_weight: Vec<f32> = Vec::new();
        for _ in 0..num_center {
            coefficients_weight.push(1.0);
            for _ in 0..num_cols {
                coefficients_centers.push(1.0);
            }
        }

        RBFRegression {
            num_center,
            centers: DenseMatrix::from_array(num_center, num_cols, &coefficients_centers),
            beta,
            weight: DenseMatrix::from_array(num_center, 1, &coefficients_weight),
            type_factoration,
        }
    }

    pub fn fit(&mut self, x: &DenseMatrix<f32>, y: &DenseMatrix<f32>) {
        let (_, n_columns) = self.centers.shape();
        let x = expand_matrix(&x, n_columns);

        let (num_rows, num_cols) = x.shape();
        let mut index: Vec<usize> = (0..num_rows).collect();

        index.shuffle(&mut thread_rng());
        let mut count = 0;
        for i in index {
            for j in 0..num_cols {
                self.centers.set(count, j, x.get(i, j));
            }
            count = count + 1;
            if count >= self.num_center {
                break;
            }
        }

        let gradient = calculate_gradient(&x, &self.centers, &self.beta);
        match self.type_factoration {
            Some(TypeFactoration::SVD) => {
                self.weight = gradient
                    .transpose()
                    .matmul(&gradient)
                    .svd_solve_mut(gradient.transpose().matmul(&y).clone())
                    .unwrap();
            }
            Some(TypeFactoration::QR) => {
                self.weight = gradient
                    .transpose()
                    .matmul(&gradient)
                    .qr_solve_mut(gradient.transpose().matmul(&y).clone())
                    .unwrap();
            }
            Some(TypeFactoration::CHOLESKY) => {
                self.weight = gradient
                    .transpose()
                    .matmul(&gradient)
                    .qr_solve_mut(gradient.transpose().matmul(&y).clone())
                    .unwrap();
            }
            _ => {
                self.weight = gradient
                    .transpose()
                    .matmul(&gradient)
                    .svd_solve_mut(gradient.transpose().matmul(&y).clone())
                    .unwrap();
            }
        }
    }

    pub fn predict(&mut self, x: &DenseMatrix<f32>) -> DenseMatrix<f32> {
        let (_, n_columns) = self.centers.shape();
        let x = expand_matrix(&x, n_columns);

        return calculate_gradient(&x, &self.centers, &self.beta).matmul(&self.weight);
    }
}

pub fn calculate_gradient(
    x: &DenseMatrix<f32>,
    centers: &DenseMatrix<f32>,
    beta: &f32,
) -> DenseMatrix<f32> {
    let (num_row, _) = x.shape();
    let (num_center, _) = centers.shape();
    let mut gradient_vector: Vec<f32> = Vec::new();
    for i in 0..num_row {
        let x_row = x.get_row(i);
        for j in 0..num_center {
            let norm: f32 = centers
                .get_row(j)
                .sub(&x_row)
                .iter()
                .map(|r| r.powf(2.0))
                .sum();
            gradient_vector.push((-beta * norm).exp());
        }
    }

    return DenseMatrix::from_array(num_row, num_center, &gradient_vector);
}
