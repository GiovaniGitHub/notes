package com.main;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

public class SimpleLinearRegression {
    public SimpleLinearRegression(){}

    public static Map<String, Double> estimateCoef(List<Double> x, List<Double> y) {
        int n = y.size();
        Mean mean = new Mean();

        double[] yPrimitive = ArrayUtils.toPrimitive(y.toArray(new Double[y.size()]));
        double[] xPrimitive = ArrayUtils.toPrimitive(x.toArray(new Double[x.size()]));
        RealVector yRealVector = MatrixUtils.createRealVector(yPrimitive);
        RealVector xRealVector = MatrixUtils.createRealVector(xPrimitive);

        double meanY = mean.evaluate(yPrimitive);
        double meanX = mean.evaluate(xPrimitive);

        double xSS = xRealVector.dotProduct(xRealVector) - n * meanX * meanX;
        double ySS = yRealVector.dotProduct(yRealVector) - n * meanY * meanY;
        double xySS = xRealVector.dotProduct(yRealVector) - n * meanX * meanY;

        double b1 = xySS / xSS;
        double b0 = meanY - b1 * meanX;
        double r = xySS / (Math.sqrt(xSS * ySS));

        Map<String, Double> resp = new HashMap<String, Double>();

        resp.put("b1", b1);
        resp.put("b0", b0);
        resp.put("r", r);

        return resp;
    }
}
