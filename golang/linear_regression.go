package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func ClassifierDataset(coefs map[int]float64, M mat.Matrix) []float64 {
	y_hat := []float64{}
	n_rows, n_cols := M.Dims()
	for i := 0; i < n_rows; i++ {
		p := 0.0
		for j := 0; j < n_cols; j++ {

			p = p + coefs[j]*M.At(i, j)
		}

		y_hat = append(y_hat, p)
	}

	return y_hat
}

func CoefEstimate(X_dense, y_dense mat.Matrix, n_cols int) map[int]float64 {
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

func LinearRegression() {
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
	coeff := CoefEstimate(X_dense, y_dense, n_cols)
	y_hat := ClassifierDataset(coeff, X_dense)

	idx := []float64{}
	for i := 0; i < len(y_hat); i++ {
		idx = append(idx, float64(i*2))
	}

	p := plot.New()

	p.Title.Text = fmt.Sprint("Linear Regression \n R2 = ", r2(y_hat, y_dense.RawMatrix().Data))

	plotutil.AddLines(p,
		"Predicted", GeneratePoints(idx, y_hat),
		"Original", GeneratePoints(idx, y_dense.RawMatrix().Data))

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "linear_regression_golang.png"); err != nil {
		panic(err)
	}
}
