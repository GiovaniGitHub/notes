package com.main;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

public class RBFRegression {
    static class Dataset {
        double[][] X;
        double[][] Xexpanded;
        double[] y;

        public void initDataset(int nrows, int ncols) {
            X = new double[nrows][ncols];
            Xexpanded = new double[nrows][ncols + 1];
            y = new double[nrows];
        }
    }

    public static Dataset readDataset(String file, int dimension) {
        Dataset dataset = new Dataset();
        Path path = Paths.get(file);
        try (Stream<String> lines = Files.lines(path)) {
            List<Object> rows = Arrays.asList(lines.toArray());
            dataset.initDataset(rows.size() - 1, dimension - 1);
            for (int i = 1; i < rows.size(); i++) {
                String s = String.valueOf(rows.get(i));
                List<String> l = Arrays.asList(s.split(","));

                for (int j = 1; j < dimension; j++) {
                    dataset.X[i - 1][j - 1] = Math.pow(Float.parseFloat(l.get(0)), dimension - j);
                    dataset.Xexpanded[i - 1][j - 1] = Math.pow(Float.parseFloat(l.get(0)), dimension - j);
                }

                dataset.Xexpanded[i - 1][dimension - 1] = 1.0;
                dataset.y[i - 1] = Float.parseFloat(l.get(l.size() - 1));
            }

        } catch (IOException e) {
            System.out.println("getMessage(): " + e.getMessage());
        }

        return dataset;
    }

    private RealMatrix centers;
    private Integer nCenters;
    private Double beta;
    private RealVector weights;

    public RBFRegression(Integer nCenters, Double beta) {
        this.nCenters = nCenters;
        this.beta = beta;
    }

    public RealMatrix getCenters() {
        return this.centers;
    }

    public void setCenters(RealMatrix centers) {
        this.centers = centers;
    }

    public void setNCenters(Integer nCenters) {
        this.nCenters = nCenters;
    }

    public void setBeta(Double beta) {
        this.beta = beta;
    }

    public void setWeights(RealVector weights) {
        this.weights = weights;
    }

    protected Double radialBasisFunction(RealVector realVector, Double beta) {
        return Math.exp(-beta * Math.pow(realVector.getNorm(), 2));
    }

    private RealMatrix getGradient(RealMatrix X) {
        double[][] gradient = new double[this.nCenters][X.getRowDimension()];
        for (int i = 0; i < this.nCenters; i++) {
            for (int j = 0; j < X.getRowDimension(); j++) {
                gradient[i][j] = this.radialBasisFunction(
                        this.centers.getRowVector(i).subtract(X.getRowVector(j)), this.beta);
            }
        }

        return (new Array2DRowRealMatrix(gradient, false)).transpose();

    }

    public void fit(RealMatrix X, RealVector y) {
        this.setCenters(Utils.getSample(X, this.nCenters));
        RealMatrix gradient = this.getGradient(X);
        RealVector weights = (new SingularValueDecomposition(gradient)).getSolver().solve(y);
        this.setWeights(weights);

    }

    public RealVector predict(RealMatrix X){
        RealMatrix gradient = this.getGradient(X);
        return gradient.operate(this.weights);
    }
}