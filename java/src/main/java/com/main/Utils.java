package com.main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class Utils {
    public static RealMatrix randomizeMatrix(RealMatrix X){
        double[][] data = X.getData();
        List<double[]> asList = Arrays.asList(data);
        Collections.shuffle(asList);
        data = asList.toArray(new double[0][0]);

        return new Array2DRowRealMatrix(data, false);
    }

    public static RealMatrix getSample(RealMatrix X, Integer n) {
        RealMatrix XRandomized = Utils.randomizeMatrix(X);
        ArrayList<double[]> asList = new ArrayList<double[]>();
        for(int i=0; i< n; i++){
            asList.add(XRandomized.getRow(i));
        }

        return new Array2DRowRealMatrix(asList.toArray(new double[0][0]), false);
    }

    public static int[] shapeMatrix(RealMatrix X){
        int[] shape = {X.getRowDimension(), X.getColumnDimension()};
        return shape;
    }
}
