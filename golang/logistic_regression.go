package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func LogisticRegression() {
	csvfile, err := os.Open("../dataset/logistic_regression.csv")
	if err != nil {
		log.Fatal(err)
	}
	lines, err := csv.NewReader(csvfile).ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	n_rows := len(lines)
	n_cols := len(lines[1])

	y_dense := mat.NewDense(n_rows-1, 1, nil)
	X_dense := mat.NewDense(n_rows-1, n_cols, nil)

	for i := 1; i < n_rows; i++ {
		value, _ := strconv.ParseFloat(lines[i][n_cols-1], 64)
		y_dense.Set(i-1, 0, value)
		for j := 0; j < n_cols; j++ {
			if j == n_cols-1 {
				X_dense.Set(i-1, n_cols-1, 1)
			} else {
				value, _ = strconv.ParseFloat(lines[i][j], 64)
				X_dense.Set(i-1, j, value)
			}
		}
	}
}
