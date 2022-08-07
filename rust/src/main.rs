use rust_regressions::clusters::knn::{KNN};
use rust_regressions::regressions::{
    linear_regression::LinearRegression, polynomial_regression::PolynomialRegression,
    rbf_regression::RBFRegression, simple_linear_regression::SimpleLinearRegression,
};
use rust_regressions::utils::io::{line_and_scatter_plot, parse_csv};
use rust_regressions::utils::types::{TypeFactoration, TypeRegression};

use smartcore::linalg::naive::dense_matrix::DenseMatrix;

use smartcore::linalg::BaseMatrix;

use std::env;
use std::{fs::File, io::BufReader};

static MSG: &str = "cargo run linear|simple|poly linear_regression|simple_linear_regression|polynomial_regression_data";

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 3, "{}", MSG);

    let dataset_name_file = &args[2];
    let type_regression = &args[1];

    if "simple" == type_regression {
        let file: File = File::open(format!("../dataset/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let dense_matrix = DenseMatrix::from_vec(tuple_result.0, tuple_result.1, &tuple_result.2);

        let y = dense_matrix.get_col_as_vec(tuple_result.1 - 1);
        let x = dense_matrix.get_col_as_vec(1);

        let mut model = SimpleLinearRegression::new();
        model.fit(&x, &y);
        let y_predictions: Vec<f32> = model.predict_list(&x);
        line_and_scatter_plot(x, vec![y, y_predictions], vec!["original", "predicted"]);
    } else if type_regression == "linear" {
        let file: File = File::open(format!("../dataset/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let dense_matrix = DenseMatrix::from_vec(tuple_result.0, tuple_result.1, &tuple_result.2);

        let y = dense_matrix.slice(0..tuple_result.0, tuple_result.1 - 1..tuple_result.1);
        let x = dense_matrix.slice(0..tuple_result.0, 1..2);

        let mut model = LinearRegression::new();
        model.fit(&x, &y);
        let y_predictions = model.predict(&x);

        let v: Vec<f32> = (0..x.shape().0).map(|v| v as f32).collect();
        line_and_scatter_plot(
            v,
            vec![y.to_row_vector(), y_predictions.to_row_vector()],
            vec!["original", "predicted"],
        );
    } else if type_regression == "poly" {
        let file: File = File::open(format!("../dataset/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let dense_matrix = DenseMatrix::from_vec(tuple_result.0, tuple_result.1, &tuple_result.2);

        let y = dense_matrix.slice(0..tuple_result.0, tuple_result.1 - 1..tuple_result.1);
        let x = dense_matrix.slice(0..tuple_result.0, 0..1);

        let mut polynomial_regression = PolynomialRegression::new(8, TypeRegression::MSE);
        polynomial_regression.fit(&x, &y, 2000, 0.08);
        let y_hat_mse = polynomial_regression.predict(&x);

        let mut polynomial_regression = PolynomialRegression::new(8, TypeRegression::MAE);
        polynomial_regression.fit(&x, &y, 2000, 0.08);
        let y_hat_mae = polynomial_regression.predict(&x);

        let mut polynomial_regression = PolynomialRegression::new(8, TypeRegression::HUBER);
        polynomial_regression.fit(&x, &y, 2000, 0.08);
        let y_hat_huber = polynomial_regression.predict(&x);

        let mut rbf = RBFRegression::new(4.0, 22, 8, None);
        rbf.fit(&x, &y);

        let y_hat_rbf = rbf.predict(&x);

        line_and_scatter_plot(
            x.clone().to_row_vector(),
            vec![
                y.to_row_vector(),
                y_hat_mse.transpose().to_row_vector(),
                y_hat_mae.transpose().to_row_vector(),
                y_hat_huber.transpose().to_row_vector(),
                y_hat_rbf.transpose().to_row_vector(),
            ],
            vec![
                "original",
                "predicted MSE",
                "predicted MAE",
                "predicted HUE",
                "predicted RBF",
            ],
        )
    } else if type_regression == "rbf" {
        let file: File = File::open(format!("../dataset/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let dense_matrix = DenseMatrix::from_vec(tuple_result.0, tuple_result.1, &tuple_result.2);

        let y = dense_matrix.slice(0..tuple_result.0, tuple_result.1 - 1..tuple_result.1);
        let x = dense_matrix.slice(0..tuple_result.0, 0..1);

        let mut rbf = RBFRegression::new(4.0, 22, 8, Some(TypeFactoration::SVD));
        rbf.fit(&x, &y);
        let y_hat_svd = rbf.predict(&x);

        let mut rbf = RBFRegression::new(4.0, 22, 8, Some(TypeFactoration::CHOLESKY));
        rbf.fit(&x, &y);
        let y_hat_cholesky = rbf.predict(&x);

        let mut rbf = RBFRegression::new(4.0, 22, 8, Some(TypeFactoration::QR));
        rbf.fit(&x, &y);
        let y_hat_qr = rbf.predict(&x);

        line_and_scatter_plot(
            x.clone().to_row_vector(),
            vec![
                y.to_row_vector(),
                y_hat_svd.transpose().to_row_vector(),
                y_hat_cholesky.transpose().to_row_vector(),
                y_hat_qr.transpose().to_row_vector(),
            ],
            vec![
                "original",
                "predicted SVD",
                "predicted CHOLESKY",
                "predicted QR",
            ],
        )
    } else if type_regression == "knn" {
        let file: File = File::open(format!("../dataset/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let dense_matrix = DenseMatrix::from_vec(tuple_result.0, tuple_result.1, &tuple_result.2);

        let y = dense_matrix.slice(10..tuple_result.0, tuple_result.1 - 1..tuple_result.1);
        let x = dense_matrix.slice(10..tuple_result.0, 1..2);

        let y_test = dense_matrix.slice(0..10, tuple_result.1 - 1..tuple_result.1);
        let x_test = dense_matrix.slice(0..10, 1..2);

        let mut model = KNN::new(x, y, 5);
        for i in 0..10 {
            let class = model.predict(x_test.get_row(i));
            println!("{:?}", (class, y_test.get_row(i)[0]));
        }
    }

    Ok(())
}
