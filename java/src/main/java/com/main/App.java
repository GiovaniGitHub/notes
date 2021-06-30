package com.main;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public final class App {
    private App() {}

    public static void main(String[] args) throws IOException {
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
        }

        Map<String, Double> m = SimpleLinearRegression.estimateCoef(dataset.get("income"), dataset.get("happiness"));
        double[] yHat = new double[dataset.get("income").size()];
        for (int i = 0; i < dataset.get("income").size(); i++) {
            yHat[i] = dataset.get("income").get(i) * m.get("b1") + m.get("b0");
        }

    }
}
