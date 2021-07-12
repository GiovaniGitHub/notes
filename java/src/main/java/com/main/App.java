package com.main;

import java.util.Map;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;

import javafx.stage.Stage;

public class App extends Application {

    private ScatterChart<Number, Number> createScatterChart(double[] y, double[] yHat){
        final ScatterChart<Number, Number> sc = new ScatterChart<>(new NumberAxis(), new NumberAxis());
        sc.setTitle("Simple Linear Regression");
        XYChart.Series series1 = new XYChart.Series();
        XYChart.Series series2 = new XYChart.Series();
        series1.setName("Original Data");
        series2.setName("Regression");
        for (int i = 0; i < y.length; i++) {
            series1.getData().add(new XYChart.Data(i, y[i]));
            series2.getData().add(new XYChart.Data(i, yHat[i]));
        }
    
        sc.getData().addAll(series1, series2);

        return sc;

    }
    @Override
    public void start(Stage stage) throws Exception {
        LinearRegression.Dataset dataset = LinearRegression.readDataset(true);
        double[] coefs = LinearRegression.estimateCoef(dataset.X, dataset.y, "");
        double[] yHat = LinearRegression.predict(dataset, coefs);

        // SimpleLinearRegression.Dataset dataset = SimpleLinearRegression.readDataset();
        // Map<String, Double> m = SimpleLinearRegression.estimateCoef(dataset.x, dataset.y);
        // double[] yHat = SimpleLinearRegression.predict(dataset, m);

        Scene scene = new Scene(createScatterChart(dataset.y, yHat));
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        Application.launch(args);
    }
}