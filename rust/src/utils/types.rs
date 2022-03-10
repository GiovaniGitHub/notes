pub enum TypeRegression {
    MAE,
    MSE,
    HUBER
}

pub enum TypeFactoration {
    SVD,
    QR,
    CHOLESKY,
}

pub enum Option<TypeFactoration>{
    None,
    Some(TypeFactoration)

}