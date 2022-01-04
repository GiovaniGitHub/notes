use rust_regressions::regressions::linear_regression::LinearRegression;
use rust_regressions::regressions::polynomial_regression::PolynomialRegression;
use rust_regressions::regressions::simple_linear_regression::SimpleLinearRegression;
use rust_regressions::utils::io::{line_and_scatter_plot, parse_csv};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use std::env;
use std::{fs::File, io::BufReader};

static MSG: &str = "cargo run linear|simple linear_regression|simple_linear_regression";

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
        let x_clone = x.clone();
        // let expanded_matrix = expand_matrix(&x, 4);

        // let (nrows, ncols) = expanded_matrix.shape();
        // let mut rng = rand::thread_rng();
        // let mut coefficients: Vec<f32> = Vec::new();
        // for _ in 0..ncols {
        //     coefficients.push(rng.gen::<f32>());
        // }

        // let mut b: f32 = 1.0;
        // let mut coef_dense = DenseMatrix::from_array(ncols, 1, &coefficients);
        // let mut y_hat = expanded_matrix.matmul(&coef_dense);
        // y_hat.add_mut(&DenseMatrix::from_vec(
        //     expanded_matrix.shape().0,
        //     1,
        //     &vec![b; expanded_matrix.shape().0],
        // ));

        // for _ in 0..2000 {
        //     let (dw, db) = update_weights_mse(&expanded_matrix, &y, &y_hat, 0.8);
        //     coef_dense.add_mut(&dw);
        //     b += db;

        //     y_hat = expanded_matrix.matmul(&coef_dense);
        //     y_hat.add_mut(&DenseMatrix::from_vec(
        //         expanded_matrix.shape().0,
        //         1,
        //         &vec![b; expanded_matrix.shape().0],
        //     ));
        // }
        let mut polynomial_regression = PolynomialRegression::new(7);
        polynomial_regression.fit(&x, &y, 1000, 0.8);
        let y_hat = polynomial_regression.predict(&x);
        let y_hat_vector = y_hat.transpose().to_row_vector();
        let y_vector = y.to_row_vector();
        let x_vector = x_clone.to_row_vector();
        line_and_scatter_plot(x_vector, vec![y_vector, y_hat_vector], vec!["original", "predicted"])
    }
    Ok(())
}
