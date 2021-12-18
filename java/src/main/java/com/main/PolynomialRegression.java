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
        double[][] Xexpanded;
        double[] y;

        public void initDataset(int nrows, int ncols){
            X = new double[nrows][ncols];
            Xexpanded = new double[nrows][ncols+1];
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
                    dataset.X[i-1][j-1] = Math.pow(Float.parseFloat(l.get(0)),dimension - j);
                    dataset.Xexpanded[i-1][j-1] = Math.pow(Float.parseFloat(l.get(0)),dimension - j);
                }

                dataset.Xexpanded[i-1][dimension-1] = 1.0;
                dataset.y[i-1] = Float.parseFloat(l.get(l.size()-1));
            }
            
        } catch (IOException e) {
            System.out.println("getMessage(): " + e.getMessage());
        }
    
            return dataset;
    }

    public static double[] estimateCoef(Dataset dataset, String type){
        RealMatrix matrix = new Array2DRowRealMatrix(dataset.Xexpanded, false);
        RealVector yMatrix = new ArrayRealVector(dataset.y, false);
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
    
    public static double[] predict(double[][] X, double[] coefs, Double bias){
        double[] yHat = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            for(int j=0; j < X[0].length;j++){
                yHat[i]+=coefs[j]*X[i][j];
            }
            if(bias != null){
                yHat[i]+= bias;
            }
        }
        return yHat;
    }

    public static List<Object> estimateCoefByGradient(Dataset dataset, double lr, int epochs, String gradientType){
        double[] coefs = new double[dataset.X[0].length];
        double bias = 0.0;
        RealMatrix xMatrix = new Array2DRowRealMatrix(dataset.X, false);
        RealVector yVector = new ArrayRealVector(dataset.y, false);
        RealVector yHatVector;
        RealVector diffVector;
        double[] dw = new double[coefs.length]; 
        double db = 0.0;
        while(epochs>0){
            yHatVector = new ArrayRealVector(predict(dataset.X, coefs, bias),false);
            diffVector = yHatVector.subtract(yVector);
            List<Object> valuesUpdate = Gradient.evaluate(xMatrix, diffVector, gradientType, 1);
            dw = (double[]) valuesUpdate.get(0);
            db = (double) valuesUpdate.get(1);
            for(int i=0; i<coefs.length; i++){
                coefs[i]-=lr*dw[i];
            }
            bias -= lr*db; 
            epochs--;
            
        }
        return Arrays.asList(coefs,bias);
    }
}

