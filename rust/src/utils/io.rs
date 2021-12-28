use nalgebra::{DMatrix, Scalar};
use std::{io::BufRead, str::FromStr};
use plotly::common::Mode;
use plotly::{Plot, Scatter};

#[allow(non_camel_case_types)]
pub fn parse_csv<f32, R>(input: R) -> Result<DMatrix<f32>, Box<dyn std::error::Error>>
where
    f32: FromStr + Scalar,
    f32::Err: std::error::Error,
    R: BufRead,
{
    let mut data = Vec::new();

    let mut rows = 0;

    for line in input.lines().skip(1) {
        rows += 1;
        for datum in line?.split_terminator(",") {
            let value: f32 = datum.trim().parse().unwrap();
            data.push(value);
        }
    }

    let cols = data.len() / rows;

    Ok(DMatrix::from_row_slice(rows, cols, &data[..]))
}

pub fn line_and_scatter_plot(x: Vec<f32>, y: Vec<Vec<f32>>, names: Vec<&str>) {
    let mut plot = Plot::new();
    let mut idx: usize = 0;

    for col in y {
        plot.add_trace(
            Scatter::new(x.clone(), col)
                .name(names[idx])
                .mode(Mode::Markers),
        );
        idx = idx + 1;
    }
    plot.show();
}