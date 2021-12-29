use smartcore::linalg::{naive::dense_matrix::DenseMatrix, svd::SVDDecomposableMatrix, BaseMatrix};

pub struct LinearRegression {
    pub coefficients: Option<DenseMatrix<f32>>,
    pub bias: Option<f32>,
}

impl LinearRegression {

    pub fn new() -> LinearRegression {
        LinearRegression {
            coefficients: None,
            bias: None,
        }
    }

    pub fn fit(&mut self, x: &DenseMatrix<f32>, y: &DenseMatrix<f32>) {
        let (nrows, num_attributes) = x.shape();
        let b = DenseMatrix::from_vec(nrows ,1, &vec![1.;nrows]);
        let a: DenseMatrix<f32> = x.h_stack(&b);
        let svd_result = a.svd_solve_mut(y.clone()).unwrap();
        
        self.coefficients = Some(svd_result.slice(0..num_attributes, 0..1));
        self.bias = Some(svd_result.get(num_attributes,0));

    }

    pub fn predict(&self, x: &DenseMatrix<f32>) -> DenseMatrix<f32>{
        let values = self.coefficients.as_ref().unwrap().clone();
        let mut y_hat = x.matmul(&values);

        y_hat.add_mut(
            &DenseMatrix::from_vec(
             x.shape().0,
             1,
             &vec![self.bias.unwrap(); x.shape().0]));

        return y_hat;
    }

}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}