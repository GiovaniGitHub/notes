package com.main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Gradient {

    public static List<Object> evaluate(RealMatrix xMatrix, RealVector diffVector, String type, double delta){
        List<Object> resp;
        switch(type){
            case "mse":
                resp = updateWeightsMse(xMatrix, diffVector);
                break;

            case "mae":
                resp = updateWeightsMae(xMatrix, diffVector);
                break;

            case "hue":
                resp = updateWeightsHuber(xMatrix, diffVector, delta);

            default:
                resp = updateWeightsMse(xMatrix, diffVector);
                break;
        }
        return resp;
    }
    private static List<Object> updateWeightsMae(RealMatrix xMatrix, RealVector diffVector){
        
        double[] dw = new double[xMatrix.getColumnDimension()]; 
        double db = 0.0;

        for(int i = 0; i<xMatrix.getColumnDimension();i++){
            dw[i] = (1.0/diffVector.getL1Norm())*xMatrix.getColumnVector(i).dotProduct(diffVector);
        }
        db = (1.0/diffVector.getL1Norm())*Arrays.stream(diffVector.toArray()).sum();
        List<Object> resp = new ArrayList<Object>();
        resp.add(dw);resp.add(db);
        return resp;
    }

    private static List<Object> updateWeightsMse(RealMatrix xMatrix, RealVector diffVector){
        double[] dw = new double[xMatrix.getColumnDimension()]; 
        double db = 0.0;
        for(int i = 0; i<xMatrix.getColumnDimension();i++){
            dw[i] = (1.0/(2*xMatrix.getColumnDimension()))*xMatrix.getColumnVector(i).dotProduct(diffVector);
        }
        db = (1.0/(2*xMatrix.getColumnDimension()))*Arrays.stream(diffVector.toArray()).sum();
        List<Object> resp = new ArrayList<Object>();
        resp.add(dw);resp.add(db);
        return resp;
    }

    private static List<Object> updateWeightsHuber(RealMatrix xMatrix, RealVector diffVector, Double delta){
        double[] dw = new double[xMatrix.getColumnDimension()]; 
        double db = 0.0;
        if(Arrays.stream(diffVector.toArray()).sum()<= delta){
            for(int i = 0; i<xMatrix.getColumnDimension();i++){
                dw[i] = (1.0/(xMatrix.getColumnDimension()))*xMatrix.getColumnVector(i).dotProduct(diffVector);
            }
        } else {
            for(int i = 0; i<xMatrix.getColumnDimension();i++){
                dw[i] = (1.0/diffVector.getL1Norm())*xMatrix.getColumnVector(i).dotProduct(diffVector);
            }
            db = (1.0/diffVector.getL1Norm())*Arrays.stream(diffVector.toArray()).sum();
        }

        List<Object> resp = new ArrayList<Object>();
        resp.add(dw);resp.add(db);
        return resp;

    }
}
