package main

import (
	"gonum.org/v1/gonum/mat"
)

func UpdateWeightsMSE(x mat.Matrix, y mat.Matrix, y_hat mat.Matrix) ([]float64, float64) {
	n_rows, n_cols := x.Dims()

	dw := []float64{}
	db := 0.0

	for j := 0; j < n_cols; j++ {
		dw = append(dw, 0)
	}

	for i := 0; i < n_rows; i++ {
		err_value := y_hat.At(i, 0) - y.At(i, 0)
		for j := 0; j < n_cols; j++ {
			dw[j] += (1.0 / (2.0 * float64(n_cols))) * x.At(i, j) * err_value
		}
		db += (1.0 / (2.0 * float64(n_cols))) * err_value
	}

	return dw, db
}

func AdjustWeight(X mat.Matrix, y mat.Matrix, w []float64, b float64, epochs int, losses []float64, lr float64, func_adjust func(mat.Matrix, mat.Matrix, mat.Matrix) ([]float64, float64), is_stochastic bool) ([]float64, float64) {
	for i := 0; i < epochs; i++ {
		y_hat := Predict(w, X, b)
		y_hat_dense := mat.NewDense(len(y_hat), 1, y_hat)
		dw, db := func_adjust(X, y, y_hat_dense)
		for j := 0; j < len(dw); j++ {
			w[j] -= dw[j] * lr
		}
		b -= lr * db
	}
	return w, b
}

func Predict(w []float64, M mat.Matrix, b float64) []float64 {
	y_hat := []float64{}
	n_rows, n_cols := M.Dims()
	for i := 0; i < n_rows; i++ {
		p := 0.0
		for j := 0; j < n_cols; j++ {
			p += w[j] * M.At(i, j)
		}
		p += b
		y_hat = append(y_hat, p)
	}

	return y_hat
}
