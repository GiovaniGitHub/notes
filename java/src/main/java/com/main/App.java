package com.main;

import java.util.Map;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;

import javafx.stage.Stage;

public class App extends Application {

    private ScatterChart<Number, Number> createScatterChart(SimpleLinearRegression.Dataset dataset, double[] yHat){
        final ScatterChart<Number, Number> sc = new ScatterChart<>(new NumberAxis(), new NumberAxis());
        sc.setTitle("Simple Linear Regression");
        XYChart.Series series1 = new XYChart.Series();
        XYChart.Series series2 = new XYChart.Series();
        series1.setName("Original Data");
        series2.setName("Regression");
        for (int i = 0; i < dataset.x.length; i++) {
            series1.getData().add(new XYChart.Data(dataset.x[i], dataset.y[i]));
            series2.getData().add(new XYChart.Data(dataset.x[i], yHat[i]));
        }
    
        sc.getData().addAll(series1, series2);

        return sc;

    }
    @Override
    public void start(Stage stage) throws Exception {
        SimpleLinearRegression.Dataset dataset = SimpleLinearRegression.readDataset();
        Map<String, Double> m = SimpleLinearRegression.estimateCoef(dataset.x, dataset.y);
        double[] yHat = SimpleLinearRegression.predict(dataset, m);

        Scene scene = new Scene(createScatterChart(dataset, yHat));
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        Application.launch(args);
    }
}