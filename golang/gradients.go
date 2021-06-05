package main

import (
	"gonum.org/v1/gonum/mat"
)

func UpdateWeightsMAE(x mat.Matrix, y mat.Matrix, y_hat mat.Matrix) ([]float64, float64) {
	n_rows, n_cols := x.Dims()

	dw := []float64{}
	err_vector := []float64{}

	for j := 0; j < n_cols; j++ {
		dw = append(dw, 0)
	}

	for i := 0; i < n_rows; i++ {
		err_vector = append(err_vector, y_hat.At(i, 0)-y.At(i, 0))
	}

	diff := []float64{}
	for j := 0; j < n_cols; j++ {
		temp := 0.0
		for i := 0; i < n_rows; i++ {
			temp += x.At(i, j) * err_vector[i]
		}
		diff = append(diff, temp)
	}

	for j := 0; j < n_cols; j++ {
		for i := 0; i < n_rows; i++ {
			dw[j] += x.At(i, j) * err_vector[i]
		}
		dw[j] = (1.0 / SumAbs(diff)) * diff[j]
	}

	db := (1.0 / SumAbs(diff)) * Sum(err_vector)

	return dw, db
}

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

func UpdateWeightsHuber(x mat.Matrix, y mat.Matrix, y_hat mat.Matrix, delta float64) ([]float64, float64) {
	n_rows, _ := x.Dims()
	dw := []float64{}
	db := 0.0

	diff := []float64{}
	for i := 0; i < n_rows; i++ {
		diff = append(diff, y_hat.At(i, 0)-y.At(i, 0))
	}

	if Sum(diff) <= delta {
		dw, db = UpdateWeightsMSE(x, y, y_hat)

	} else {
		dw, db = UpdateWeightsMAE(x, y, y_hat)
		db = delta * db
		for i := 0; i < len(dw); i++ {
			dw[i] = dw[i] * delta
		}
	}
	return dw, db
}

func AdjustWeight(p Parameters) ([]float64, float64) {

	if p.is_stochastic {
		p.X, p.y = RandomizeDataset(p.X, p.y)
	}

	for i := 0; i < p.epochs; i++ {
		y_hat := Predict(p.w, p.X, p.b)
		y_hat_dense := mat.NewDense(len(y_hat), 1, y_hat)

		dw := []float64{}
		db := 0.0

		switch func_type := p.func_type; func_type {

		case "mse":
			dw, db = UpdateWeightsMSE(p.X, p.y, y_hat_dense)
		case "mae":
			dw, db = UpdateWeightsMAE(p.X, p.y, y_hat_dense)
		case "huber":
			dw, db = UpdateWeightsHuber(p.X, p.y, y_hat_dense, p.delta)

		}
		for j := 0; j < len(dw); j++ {
			p.w[j] -= dw[j] * p.lr
		}
		p.b -= p.lr * db
	}
	return p.w, p.b
}
