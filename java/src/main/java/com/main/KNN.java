package com.main;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class KNN{
    static class Dataset{
        double[][] X;
        double[] y;

        public void initDataset(int nrows, int ncols){
            X = new double[nrows][ncols];
            y = new double[nrows];
        }
    }

    public static Dataset readDataset(String file, int dimension) {
        Dataset dataset = new Dataset();
        Path path = Paths.get(file);
        try (Stream<String> lines = Files.lines(path)) {
            List<Object> rows = Arrays.asList(lines.toArray());
            dataset.initDataset(rows.size()-1, dimension-1);
            for(int i = 1; i<rows.size(); i++){
                String s = String.valueOf(rows.get(i));
                List<String> l = Arrays.asList(s.split(","));

                for(int j=1; j<dimension; j++){
                    dataset.X[i-1][j-1] = Float.parseFloat(l.get(j));
                }

                dataset.y[i-1] = Float.parseFloat(l.get(l.size()-1));
            }
            
        } catch (IOException e) {
            System.out.println("getMessage(): " + e.getMessage());
        }
    
            return dataset;
    }

    public static double predict(Dataset dataset, String type){
        RealMatrix matrix = new Array2DRowRealMatrix(dataset.X, false);
        RealVector yMatrix = new ArrayRealVector(dataset.y, false);
        return 0;
    }
}