package com.main;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.summary.Sum;

public class SimpleLinearRegression {
    public SimpleLinearRegression(){}

    static class Dataset{
        double[] x;
        double[] y;

        public void initDataset(int nrows){
            x = new double[nrows];
            y = new double[nrows];
        }
    }

    public static Dataset readDataset() {
        Dataset dataset = new Dataset();
        Path path = Paths.get("../dataset/simple_regression.csv");
        try (Stream<String> lines = Files.lines(path)) {
            List<Object> rows = Arrays.asList(lines.toArray());
            dataset.initDataset(rows.size()-1);
            for(int i = 1; i<rows.size(); i++){
                String s = String.valueOf(rows.get(i));
                List<String> l = Arrays.asList(s.split(","));
                dataset.x[i-1] = Float.parseFloat(l.get(1));
                dataset.y[i-1] = Float.parseFloat(l.get(2));
            }
            
        } catch (IOException e) {
            System.out.println("getMessage(): " + e.getMessage());
        }

        return dataset;
    }

    public static Map<String, Double> estimateCoef(double[] x, double[] y) {
        int n = y.length;
        Sum sum = new Sum();

        RealVector yRealVector = MatrixUtils.createRealVector(y);
        RealVector xRealVector = MatrixUtils.createRealVector(x);

        double sumY = sum.evaluate(y);
        double sumX = sum.evaluate(x);

        double xSS = xRealVector.dotProduct(xRealVector) - (sumX * sumX)/n;
        double ySS = yRealVector.dotProduct(yRealVector) - (sumY * sumY)/n;
        double xySS = xRealVector.dotProduct(yRealVector) - (sumX * sumY)/n;

        double b1 = xySS / xSS;
        double b0 = (sumY/n) - b1 * (sumX/n);
        double r = xySS / (Math.sqrt(xSS * ySS));

        Map<String, Double> resp = new HashMap<String, Double>();

        resp.put("b1", b1);
        resp.put("b0", b0);
        resp.put("r", r);

        return resp;
    }

    public static double[] predict(Dataset dataset, Map<String, Double> m){
        double[] yHat = new double[dataset.x.length];
        for (int i = 0; i < dataset.x.length; i++) {
            yHat[i] = dataset.x[i] * m.get("b1") + m.get("b0");
        }

        return yHat;
    }
}
