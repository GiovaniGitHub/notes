use rust_regressions::regressions::linear_regression::LinearRegression;
use rust_regressions::regressions::simple_linear_regression::SimpleLinearRegression;
use rust_regressions::utils::io::{line_and_scatter_plot, parse_csv};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use std::env;
use std::{fs::File, io::BufReader};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
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
        let x = dense_matrix.slice(0..tuple_result.0,1..3);

        let mut model = LinearRegression::new();
        model.fit(&x, &y);
        let y_predictions = model.predict(&x);
        let v: Vec<f32> = (0..x.shape().0).map(|v| v as f32).collect();
        line_and_scatter_plot( v, vec![y.to_row_vector(), y_predictions.to_row_vector()], vec!["original", "predicted"]);
    }
    Ok(())
}

// use smartcore::linalg::naive::dense_matrix::*;
// use smartcore::linalg::Matrix;
// use smartcore::linalg::svd::SVDDecomposableMatrix;
// fn main(){
//     let x = DenseMatrix::from_2d_array(&[
//         &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
//         &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
//         &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
//         &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
//         &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
//         &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
//         &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
//         &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
//         &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
//         &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
//         &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
//         &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
//         &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
//         &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
//         &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
//         &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
//    ]);

//     let y: Vec<f64> = vec![83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0,
//     100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9];

//     let y_m = DenseMatrix::from_row_vector(y.clone());
//     let b = y_m.transpose();
//     let (x_nrows, num_attributes) = x.shape();
//     let (y_nrows, _) = b.shape();


//     let a = x.h_stack(&DenseMatrix::ones(x_nrows, 1));

//     let w =  a.svd_solve_mut(b).unwrap();

//     let wights = w.slice(0..num_attributes, 0..1);
//     println!("{:?}", wights);
// }