package com.main;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;


public class LinearRegression{

    static class Dataset{
        double[][] X;
        double[] y;

        public void initDataset(int nrows){
            X = new double[nrows][nrows];
            y = new double[nrows];
        }
    }
    public static Dataset readDataset() {
        Dataset dataset = new Dataset();
        Path path = Paths.get("../dataset/linear_regression.csv");
        try (Stream<String> lines = Files.lines(path)) {
            List<Object> rows = Arrays.asList(lines.toArray());
            dataset.initDataset(rows.size());
            for(int i = 1; i<rows.size(); i++){
                String s = String.valueOf(rows.get(i));
                List<String> l = Arrays.asList(s.split(","));
                dataset.X[i][0] = Float.parseFloat(l.get(0));
                dataset.X[i][1] = Float.parseFloat(l.get(1));
                dataset.y[i] = Float.parseFloat(l.get(2));
            }
            
        } catch (IOException e) {
            System.out.println("getMessage(): " + e.getMessage());
        }

        return dataset;
    }
 }