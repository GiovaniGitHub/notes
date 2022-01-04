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
