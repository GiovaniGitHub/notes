package com.main;

import java.util.List;
import java.util.Map;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import java.util.Arrays;
import javafx.stage.Stage;

public class App extends Application {

    private ScatterChart<Number, Number> createScatterChart(double[] x, double[] y, double[] yHat){
        final ScatterChart<Number, Number> sc = new ScatterChart<>(new NumberAxis(), new NumberAxis());
        sc.setTitle("Simple Linear Regression");
        XYChart.Series series1 = new XYChart.Series();
        XYChart.Series series2 = new XYChart.Series();
        series1.setName("Original Data");
        series2.setName("Regression");
        for (int i = 0; i < y.length; i++) {
            series1.getData().add(new XYChart.Data(x[i], y[i]));
            series2.getData().add(new XYChart.Data(x[i], yHat[i]));
        }
    
        sc.getData().addAll(series1, series2);

        return sc;

    }
    @Override
    public void start(Stage stage) throws Exception {
        Parameters params = getParameters();
        List<String> list = params.getRaw();
        double[] coefs = null;
        double[] yHat = null;
        double[] yAxis = null;
        double[] xAxis = null;
        switch (list.get(0)) {
            case "simple":
                SimpleLinearRegression.Dataset datasetSimple = SimpleLinearRegression.readDataset("../dataset/simple_regression.csv");
                Map<String, Double> m = SimpleLinearRegression.estimateCoef(datasetSimple.x, datasetSimple.y);
                yHat = SimpleLinearRegression.predict(datasetSimple, m);
                yAxis = datasetSimple.y;
                xAxis = datasetSimple.x;
                break;

            case "linear":
                LinearRegression.Dataset datasetLinear = LinearRegression.readDataset("../dataset/linear_regression.csv", true);
                coefs = LinearRegression.estimateCoef(datasetLinear.X, datasetLinear.y, "");
                yHat = LinearRegression.predict(datasetLinear, coefs);
                yAxis = datasetLinear.y;
                xAxis = new double[datasetLinear.y.length];
                for(int i=0;i<yAxis.length;i++){
                    xAxis[i] = i;
                }
                break;

            case "poly":
                PolynomialRegression.Dataset datasetPoly = PolynomialRegression.readDataset("../dataset/polynomial_regression_data.csv", 7);
                coefs = PolynomialRegression.estimateCoef(datasetPoly, datasetPoly.y, "");
                yHat = PolynomialRegression.predict(datasetPoly.XwithBias, coefs);
                yAxis = datasetPoly.y;
                xAxis = new double[datasetPoly.X.length];
                for(int i=0; i<yAxis.length; i++){
                    xAxis[i] = datasetPoly.X[i][datasetPoly.X[i].length-2];
                }
                break;
            default:
                System.out.println("Please run argument simple|linear|poly");
                break;
        }

        if(yHat != null && yAxis != null && xAxis != null){
            Scene scene = new Scene(createScatterChart(xAxis, yAxis, yHat));
            stage.setScene(scene);
            stage.show();
        }
    }

    public static void main(String[] args) {
        Application.launch(args);
    }
}