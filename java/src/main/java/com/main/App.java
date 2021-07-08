package com.main;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class App extends Application {

    private Map<String, List<Double>> readDataset() {
        Map<String, List<Double>> dataset = new HashMap<String, List<Double>>();
        dataset.put("income", new ArrayList<>());
        dataset.put("happiness", new ArrayList<>());
        try (Scanner scanner = new Scanner(new File("../dataset/simple_regression.csv"));) {
            scanner.nextLine();
            while (scanner.hasNextLine()) {
                List<String> l = Arrays.asList(scanner.nextLine().split("\\s*,\\s*"));
                dataset.get("income").add(Double.parseDouble(l.get(1)));
                dataset.get("happiness").add(Double.parseDouble(l.get(2)));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return dataset;
    }

    @Override
    public void start(Stage stage) throws Exception {
        Map<String, List<Double>> dataset = readDataset();
        Map<String, Double> m = SimpleLinearRegression.estimateCoef(dataset.get("income"), dataset.get("happiness"));
        double[] yHat = new double[dataset.get("income").size()];
        for (int i = 0; i < dataset.get("income").size(); i++) {
            yHat[i] = dataset.get("income").get(i) * m.get("b1") + m.get("b0");
        }

        // stage.setTitle("Simple Linear Regression");
        final ScatterChart<Number, Number> sc = new ScatterChart<>(new NumberAxis(), new NumberAxis());
        sc.setTitle("Simple Linear Regression");
        XYChart.Series series1 = new XYChart.Series();
        XYChart.Series series2 = new XYChart.Series();
        series1.setName("Original Data");
        series2.setName("Regression");
        for (int i = 0; i < dataset.get("income").size(); i++) {
            series1.getData().add(new XYChart.Data(dataset.get("income").get(i), dataset.get("happiness").get(i)));
            series2.getData().add(new XYChart.Data(dataset.get("income").get(i), yHat[i]));
        }

        sc.getData().addAll(series1, series2);
        Scene scene = new Scene(sc);
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        Application.launch(args);
    }
}