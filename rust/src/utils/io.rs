use nalgebra::Scalar;
use plotly::common::{Mode, Marker};
use plotly::{Plot, Scatter};
use std::{io::BufRead, str::FromStr};


#[allow(non_camel_case_types)]
pub fn parse_csv<f32, R>(input: R) -> Result<(usize, usize, Vec<f32>), Box<dyn std::error::Error>>
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

    Ok((rows, cols, data))
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

pub fn scatter_plot(x: Vec<f32>, y: Vec<f32>, cls: Vec<usize>) {
    let mut plot = Plot::new();
    let mut unique_values = cls.clone();
    unique_values.sort();
    unique_values.dedup();

    for value_label in unique_values {
        let label_str = format!("Cluster {}", value_label);
        let idxs: Vec<_> = cls
            .iter()
            .enumerate()
            .filter(|&(_1, _0)| _0.eq(&value_label))
            .map(|(index, _)| index)
            .collect();
        plot.add_trace(
            Scatter::new(
                x.iter()
                    .enumerate()
                    .filter(|(i, _)| idxs.contains(i))
                    .map(|(_, v)| v.clone())
                    .collect::<Vec<f32>>(),
                y.iter()
                    .enumerate()
                    .filter(|(i, _)| idxs.contains(i))
                    .map(|(_, v)| v.clone())
                    .collect::<Vec<f32>>(),
            )
            .mode(Mode::Markers)
            .name(&label_str)
            .marker(Marker::new().size(12))
        );
    }

    plot.show();
}
