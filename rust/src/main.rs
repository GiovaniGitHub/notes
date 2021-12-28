use std::{fs::File, io::BufReader};
use rust_regressions::utils::io::{parse_csv, line_and_scatter_plot};
use rust_regressions::regressions::simple_linear_regression::SimpleLinearRegression;
use nalgebra::DMatrix;

fn main() -> std::io::Result<()> {
    let file = File::open("../dataset/simple_regression.csv").unwrap();
    let bos: DMatrix<f32> = parse_csv(BufReader::new(file)).unwrap();

    let y_col = bos.column(2);
    let x_col = bos.column(1);

    let mut y: Vec<f32> = Vec::new();
    let mut x: Vec<f32> = Vec::new();

    for (idx, value) in y_col.iter().enumerate(){
        y.push(*value);
        x.push(x_col[idx]);
    }
    let mut model = SimpleLinearRegression::new();
    model.fit(&x, &y);
    let y_predictions : Vec<f32> = model.predict_list(&x);
    line_and_scatter_plot(x, vec![y, y_predictions], vec!["original", "predicted"]);
    Ok(())
}
