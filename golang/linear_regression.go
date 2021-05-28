package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func coef_estimate(X_dense, y_dense mat.Matrix, n_cols int) map[int]float64 {
	qr := new(mat.QR)
	qr.Factorize(X_dense)
	q := new(mat.Dense)
	reg := new(mat.Dense)
	qr.QTo(q)
	qr.RTo(reg)

	qtr := q.T()
	qty := new(mat.Dense)
	qty.Mul(qtr, y_dense)

	c := make([]float64, n_cols)
	for i := n_cols - 1; i >= 0; i-- {
		c[i] = qty.At(i, 0)
		for j := i + 1; j < n_cols; j++ {
			c[i] -= c[j] * reg.At(i, j)
		}
		c[i] /= reg.At(i, i)
	}

	coeff := make(map[int]float64, n_cols-1)
	for i, val := range c {
		coeff[i] = val
	}

	return coeff
}
func main() {
	csvfile, err := os.Open("../dataset/linear_regression.csv")
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
	coeff := coef_estimate(X_dense, y_dense, n_cols)
	fmt.Println(coeff)
}
