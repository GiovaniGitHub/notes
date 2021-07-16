package com.main;

import java.util.List;
import java.util.Map;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.Arrays;

public class App extends Application {

    private XYChart.Series createSeries(String name, double[] xValues, double[] yValues){
        XYChart.Series series = new XYChart.Series();
        series.setName(name);
        for (int i = 0; i < yValues.length; i++) {
            series.getData().add(new XYChart.Data(xValues[i], yValues[i]));
        }
        return series;
    }

    private ScatterChart<Number, Number> createScatterChart(double[] x, double[] y, List<double[]> yHatList, List<String> yHatNames, String title){
        final ScatterChart<Number, Number> sc = new ScatterChart<>(new NumberAxis(), new NumberAxis());
        sc.setTitle(title);
        XYChart.Series series = createSeries(title, x, y);
        sc.getData().addAll(series);
        for(int j = 0; j < yHatList.size(); j++){
            sc.getData().add(createSeries(yHatNames.get(j), x, yHatList.get(j)));
        }

        return sc;

    }
    @Override
    public void start(Stage stage) throws Exception {
        Parameters params = getParameters();
        List<String> list = params.getRaw();
        double[] coefs = null;
        double[] yHat = null;
        double[] xAxis = null;
        List<double[]> yHatList;
        List<String> yHatNames;
        Scene scene = null;
        switch (list.get(0)) {
            case "simple":
                SimpleLinearRegression.Dataset datasetSimple = SimpleLinearRegression.readDataset("../dataset/simple_regression.csv");
                Map<String, Double> m = SimpleLinearRegression.estimateCoef(datasetSimple.x, datasetSimple.y);
                yHat = SimpleLinearRegression.predict(datasetSimple, m);
                xAxis = datasetSimple.x;
                yHatList = new ArrayList<double[]>();
                yHatNames = new ArrayList<String>();
                yHatList.add(yHat);
                yHatNames.add("Predicted");
                scene = new Scene(createScatterChart(xAxis, datasetSimple.y, yHatList, yHatNames, "Simple Linear Regression"));
                stage.setScene(scene);
                break;

            case "linear":
                LinearRegression.Dataset datasetLinear = LinearRegression.readDataset("../dataset/linear_regression.csv", true);
                coefs = LinearRegression.estimateCoef(datasetLinear.X, datasetLinear.y, "");
                yHat = LinearRegression.predict(datasetLinear, coefs);
                xAxis = new double[datasetLinear.y.length];
                for(int i=0;i<datasetLinear.y.length;i++){
                    xAxis[i] = i;
                }
                yHatList = new ArrayList<double[]>();
                yHatNames = new ArrayList<String>();
                yHatList.add(yHat);
                yHatNames.add("Predicted");
                scene = new Scene(createScatterChart(xAxis, datasetLinear.y, yHatList, yHatNames, "Linear Regression"));
                stage.setScene(scene);;
                break;

            case "poly":
                PolynomialRegression.Dataset datasetPoly = PolynomialRegression.readDataset("../dataset/polynomial_regression_data.csv", 7);
                coefs = PolynomialRegression.estimateCoef(datasetPoly, "");
                yHat = PolynomialRegression.predict(datasetPoly.Xexpanded, coefs, null);
                
                List<Object> coefsAndBias = PolynomialRegression.estimateCoefByGradient(datasetPoly, 0.1 ,200, "mae");
                double[] coefsGradient = (double[]) coefsAndBias.get(0);
                double bias = (double) coefsAndBias.get(1);
                double[] yHatGradient = PolynomialRegression.predict(datasetPoly.X, coefsGradient, bias);
                xAxis = new double[datasetPoly.X.length];
                for(int i=0; i<xAxis.length; i++){
                    xAxis[i] = datasetPoly.X[i][datasetPoly.X[i].length-2];
                }
                yHatList = new ArrayList<double[]>();
                yHatNames = new ArrayList<String>();
                yHatList.add(yHat);yHatList.add(yHatGradient);
                yHatNames.add("OLS Method");yHatNames.add("Gradient Method");
                scene = new Scene(createScatterChart(xAxis, datasetPoly.y, yHatList, yHatNames, "Polynomial Regression"));
                stage.setScene(scene);
                break;

            default:
                System.out.println("Please run argument simple|linear|poly");
                break;
        }

        if(scene != null){
            stage.show();
        }
    }

    public static void main(String[] args) {
        Application.launch(args);
    }
}