package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

type RBFRegressionStruct struct {
	NumCenters   int
	Centers      mat.Matrix
	Beta         float64
	Weight       mat.Matrix
	Factoriation TypeFactoration
}

func (rbf *RBFRegressionStruct) SetCenters(X mat.Matrix) {
	_, n_cols := X.Dims()

	rand.Seed(time.Now().UTC().UnixNano())
	NewCenters := mat.NewDense(rbf.NumCenters, n_cols, nil)

	for i, value := range rand.Perm(rbf.NumCenters) {
		for j := 0; j < n_cols; j++ {
			NewCenters.Set(i, j, X.At(value, j))
		}
	}

	rbf.Centers = NewCenters
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

func (rbf *RBFRegressionStruct) CalculateGradient(X mat.Matrix) mat.Matrix {
	n_rows, _ := X.Dims()
	n_centers, _ := rbf.Centers.Dims()
	gradientArray := mat.NewDense(n_rows, n_centers, nil) //([]float64, n_rows*n_centers)
	for i := 0; i < n_rows; i++ {
		row_data := GetRowMatrix(X, i)
		for j := 0; j < n_centers; j++ {
			row_center := GetRowMatrix(rbf.Centers, j)
			gradientArray.Set(i, j, RadialBasisVector(row_data, row_center, rbf.Beta))
		}
	}

	return gradientArray
}

func (rbf *RBFRegressionStruct) Fit(X mat.Matrix, y mat.Matrix) {
	gradient := rbf.CalculateGradient(X)
	_, n_cols_gradient := gradient.Dims()

	qr := new(mat.QR)
	q := new(mat.Dense)
	reg := new(mat.Dense)

	qr.Factorize(gradient)

	qr.QTo(q)
	qr.RTo(reg)

	qtr := q.T()

	qty := new(mat.Dense)
	qty.Mul(qtr, y)

	c := make([]float64, n_cols_gradient)
	for i := n_cols_gradient - 1; i >= 0; i-- {
		c[i] = qty.At(i, 0)
		for j := i + 1; j < n_cols_gradient; j++ {
			c[i] -= c[j] * reg.At(i, j)
		}
		c[i] /= reg.At(i, i)
	}

	rbf.Weight = mat.NewDense(n_cols_gradient, 1, c)
}

func RBFRegression() {
	csvfile, err := os.Open("../dataset/polynomial_regression_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	lines, err := csv.NewReader(csvfile).ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	n_rows := len(lines)
	n_cols := 5
	y_dense := mat.NewDense(n_rows-1, 1, nil)
	X_dense := mat.NewDense(n_rows-1, n_cols, nil)

	for i := 1; i < n_rows; i++ {
		value_y, _ := strconv.ParseFloat(lines[i][1], 64)
		y_dense.Set(i-1, 0, value_y)
		for j := 0; j < n_cols; j++ {
			value_x, _ := strconv.ParseFloat(lines[i][0], 64)
			value_x = math.Pow(value_x, float64(j+1))
			X_dense.Set(i-1, j, value_x)
		}
	}

	RBF := RBFRegressionStruct{NumCenters: 20, Centers: nil, Beta: 4.0, Factoriation: SVD}
	RBF.SetCenters(X_dense)
	RBF.Fit(X_dense, y_dense)

}
