package com.main;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Stream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.MathArrays;

public class KNN {
    static class Dataset {
        double[][] X;
        double[] y;

        public void initDataset(int nrows, int ncols) {
            X = new double[nrows][ncols];
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
                    dataset.X[i - 1][j - 1] = Float.parseFloat(l.get(j));
                }

                dataset.y[i - 1] = Float.parseFloat(l.get(l.size() - 1));
            }

        } catch (IOException e) {
            System.out.println("getMessage(): " + e.getMessage());
        }

        return dataset;
    }

    public static Map<String, Dataset> getTrainTest(double[][] X, double[] y, double percent) {
        int[] shape = Utils.shapeMatrix(new Array2DRowRealMatrix(X, false));
        int sizeTraingSet = (int) Math.round(shape[0] * percent);

        ArrayList<Integer> idxArray = new ArrayList<Integer>();
        for (int i = 0; i < shape[0]; i++) {
            idxArray.add(i);
        }

        Collections.shuffle(idxArray);

        double[][] dataXTraining = new double[sizeTraingSet][shape[1]];
        double[] dataYTraining = new double[sizeTraingSet];

        double[][] dataXTesting = new double[shape[0] - sizeTraingSet][shape[1]];
        double[] dataYTesting = new double[shape[0] - sizeTraingSet];
        for (int i = 0; i < sizeTraingSet; i++) {
            for (int j = 0; j < shape[1]; j++) {
                dataXTraining[i][j] = X[idxArray.get(i)][j];
            }
            dataYTraining[i] = y[idxArray.get(i)];

        }

        for (int i = sizeTraingSet; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                dataXTesting[i - sizeTraingSet][j] = X[idxArray.get(i)][j];
            }
            dataYTesting[i - sizeTraingSet] = y[idxArray.get(i)];

        }
        Dataset trainDataset = new Dataset();
        trainDataset.initDataset(dataXTraining.length, shape[1]);
        trainDataset.X = dataXTraining;
        trainDataset.y = dataYTraining;

        Dataset testDataset = new Dataset();
        testDataset.initDataset(dataXTesting.length, shape[1]);
        testDataset.X = dataXTesting;
        testDataset.y = dataYTesting;
        HashMap<String, Dataset> resp = new HashMap<String, Dataset>();
        resp.put("datasetTrain", trainDataset);
        resp.put("datasetTest", testDataset);

        return resp;
    }

    public static Integer predict(Dataset dataset, double[] vector, int nNeighborhood) {
        double[][] distances = new double[nNeighborhood][2];
        for (int i = 0; i < nNeighborhood; i++) {
            distances[i][0] = Double.POSITIVE_INFINITY;
            distances[i][1] = 0;
        }
        for (int i = 0; i < dataset.X.length; i++) {
            double d = MathArrays.distance(dataset.X[i], vector);
            Arrays.sort(distances, Comparator.comparingDouble(row -> row[0]));
            if (distances[nNeighborhood - 1][0] > d) {
                distances[nNeighborhood - 1][0] = d;
                distances[nNeighborhood - 1][1] = dataset.y[i];
            }
        }

        Integer[] labelsNeighborhood = Arrays.stream(distances).map(x -> (int) x[1]).toArray(Integer[]::new);

        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i : labelsNeighborhood) {
            Integer count = map.get(i);
            map.put(i, count != null ? count + 1 : 1);
        }

        Integer label = Collections.max(map.entrySet(),
                new Comparator<Map.Entry<Integer, Integer>>() {
                    @Override
                    public int compare(Entry<Integer, Integer> o1, Entry<Integer, Integer> o2) {
                        return o1.getValue().compareTo(o2.getValue());
                    }
                }).getKey();
        return label;
    }
}