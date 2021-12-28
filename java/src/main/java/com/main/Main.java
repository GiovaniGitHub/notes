package com.main;

import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;


public class Main {
    public class Competitor{

      String name;
      float dificult;
      ArrayList<Float> values = new ArrayList<Float>();

      public void setName(String name){
        this.name = name;
      }
      
      public void setDificult(float dificult){
        this.dificult = dificult;
      }
      
      public void addValue(float value){
        this.values.add(value);
      }

      public float calculateFinalNote(){
        float result = 0;
        for(Float value: this.values){
          if (value < Collections.max(this.values) & value > Collections.min(this.values)){
            result += value;
          }
        }
        return result;
      }

    } 
    public static void main(String[] args) throws IOException {

      Scanner sc = new Scanner(System.in);
      Map<String, Double> map = new HashMap<String, Double>();
      System.out.print("A quantidade de competidores");
      int qtd = sc.nextInt();
      System.out.println("Digite o fator");
      double factor = sc.nextDouble();

      for (int i=0; i<qtd; i++){
        double maxValue = 0;
        double minValue = Double.MAX_VALUE;
        System.out.print("O nome do competidor: ");
        String name = sc.next();

        ArrayList<Double> arrayValues = new ArrayList<Double>();
        System.out.print("Digite as 7 notas do competidor: ");
        for(int j=0; j<7; j++){
          double value = sc.nextDouble();
          arrayValues.add(value);
          if (value < minValue){
            minValue = value;
          }
          if (value > maxValue){
            maxValue = value;
          }
          arrayValues.add(value);
          
        }
        arrayValues.remove(minValue);
        arrayValues.remove(maxValue);
        double sum = 0;
        for(int k = 0; k < arrayValues.size(); k++)
            sum += arrayValues.get(k);
        map.put(name, sum*factor);
      }

      for (Map.Entry me : map.entrySet()) {
        System.out.println("Key: "+me.getKey() + " & Value: " + me.getValue());
      }
      sc.close();
    }

  }