use smartcore::linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix, BaseVector};

use crate::utils::types::TypeRegression;
use rand::Rng;
pub struct RBFRegression {
    pub num_center: usize,
    pub centers: DenseMatrix<f32>,
    pub beta: f32,
    pub weight: DenseMatrix<f32>,
    pub type_regression: TypeRegression,
}

impl RBFRegression {
    pub fn new(
        beta: f32,
        num_center: usize,
        num_cols: usize,
        type_regression: TypeRegression,
    ) -> RBFRegression {
        let mut rng = rand::thread_rng();
        let mut coefficients_centers: Vec<f32> = Vec::new();
        let mut coefficients_weight: Vec<f32> = Vec::new();
        for _ in 0..num_center {
            coefficients_weight.push(rng.gen::<f32>());
            for _ in 0..num_cols {
                coefficients_centers.push(rng.gen::<f32>());
            }
        }

        RBFRegression {
            num_center,
            centers: DenseMatrix::from_array(num_center, num_cols, &coefficients_centers),
            beta,
            weight: DenseMatrix::from_array(num_center, 1, &coefficients_weight),
            type_regression,
        }
    }

    fn calculate_gradient(self, x: DenseMatrix<f32>) -> DenseMatrix<f32> {
        let (num_row, num_cols) = x.shape();
        let mut gradient_vector: Vec<f32> = Vec::new();
        for i in 0..num_row {
            let x_row = x.get_row(i);
            for j in 0..self.num_center {
                let norm: f32 = self
                    .centers
                    .get_row(j)
                    .sub(&x_row)
                    .iter()
                    .map(|x| x.powf(2.0))
                    .sum();
                gradient_vector.push((-self.beta * norm).exp());
            }
        }

        let M: DenseMatrix<f32> = DenseMatrix::from_array(num_row, num_cols, &gradient_vector);
        return M.transpose();
    }

    fn fit(self, x: DenseMatrix<f32>, y: Vec<f32>) {
        todo!("Fit Method");
    }
}
