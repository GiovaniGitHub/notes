package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type RBFRegression struct {
	NumCenters   uint
	Centers      mat.Matrix
	Beta         float32
	Weight       mat.Matrix
	Factoriation TypeFactoration
}

func RadialBasisVector(center mat.Matrix, vector mat.Matrix, beta float64) float64 {
	var norm float64
	n_rows_center, n_cols_center := center.Dims()
	n_rows_vector, n_cols_vector := vector.Dims()

	if n_rows_center != n_rows_vector || n_cols_center != n_cols_vector {
		fmt.Errorf("Need same dimension")
	}
	for i := 0; i < n_rows_center; i++ {
		for j := 0; j < n_cols_center; j++ {
			norm += math.Pow(center.At(i, j)-vector.At(i, j), 2)
		}
	}
	return math.Exp(-beta * math.Sqrt(norm))
}

func CalculateGradient(X mat.Matrix, Centers mat.Matrix, beta float64) mat.Matrix {
	n_rows, _ := X.Dims()
	n_centers, _ := Centers.Dims()
	gradientArray := make([]float64, n_rows*n_centers)
	for i := 0; i < n_rows; i++ {
		row_data := GetRowMatrix(X, i)
		for j := 0; j < n_centers; j++ {
			row_center := GetRowMatrix(Centers, j)
			gradientArray = append(gradientArray, RadialBasisVector(row_data, row_center, beta))
		}
	}

	return mat.NewDense(n_rows, n_centers, gradientArray)
}
