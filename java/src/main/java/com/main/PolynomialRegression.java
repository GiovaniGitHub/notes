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
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

public class PolynomialRegression {
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
            dataset.initDataset(rows.size(), dimension);
            for(int i = 1; i<rows.size(); i++){
                String s = String.valueOf(rows.get(i));
                List<String> l = Arrays.asList(s.split(","));
                for(int j=0;j<dimension;j++){
                    dataset.X[i][j] = Math.pow(Float.parseFloat(l.get(0)),dimension-(j+1));
                }

                dataset.y[i] = Float.parseFloat(l.get(l.size()-1));
            }
            
        } catch (IOException e) {
            System.out.println("getMessage(): " + e.getMessage());
        }
    
            return dataset;
        }

        public static double[] estimateCoef(double[][] X, double[] y, String type){
            RealMatrix matrix = new Array2DRowRealMatrix(X, false);
            RealVector yMatrix = new ArrayRealVector(y, false);
            RealVector result;
            switch (type) {
                case "svc":
                    result = (new SingularValueDecomposition(matrix)).getSolver().solve(yMatrix);
                    break;
            
                default:
                    result = (new QRDecomposition(matrix)).getSolver().solve(yMatrix);
                    break;
            }
            return result.toArray();  
        }
    
        public static double[] predict(Dataset dataset, double[] coefs){
            double[] yHat = new double[dataset.X.length];
            for (int i = 0; i < dataset.X.length; i++) {
                yHat[i] = coefs[coefs.length-1];
                for(int j=0; j < coefs.length-1;j++){
                    yHat[i]+=coefs[j]*dataset.X[i][j];
                }
            }
            return yHat;
        }
}

