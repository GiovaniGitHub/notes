use crate::utils::stats;

pub struct OLSRegression {
    pub coefficient: Option<f32>,
    pub bias: Option<f32>,
}

impl OLSRegression {
    pub fn fit(&mut self, x_values: &Vec<f32>, y_values: &Vec<f32>) {
        let b1 = stats::covariance(x_values, y_values) / stats::variance(x_values);
        self.bias = Some(stats::mean(y_values) - b1 * stats::mean(x_values));
        self.coefficient = Some(b1);
    }

    pub fn new() -> OLSRegression {
        OLSRegression {
            coefficient: None,
            bias: None,
        }
    }

    pub fn predict(&self, x: f32) -> f32 {
        if self.coefficient.is_none() || self.bias.is_none() {
            panic!("fit(..) must be called first");
        }

        let b0 = self.bias.unwrap();
        let b1 = self.coefficient.unwrap();
        
        return b0 + b1 * x;
    }

    pub fn predict_list(&self, x_values: &Vec<f32>) -> Vec<f32> {
        let mut predictions = Vec::new();

        for i in 0..x_values.len() {
            predictions.push(self.predict(x_values[i]));
        }

        return predictions;
    }
}