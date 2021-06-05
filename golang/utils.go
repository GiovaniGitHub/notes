package main

import (
	"math"
	"math/rand"
	"time"

	"github.com/montanaflynn/stats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot/plotter"
)

func GeneratePoints(x []float64, y []float64) plotter.XYs {
	pts := make(plotter.XYs, len(y))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}
	return pts
}

func r2(y_hat []float64, y []float64) float64 {
	SSres := 0.0
	SStol := 0.0
	yMean, _ := stats.Mean(y)
	for i := 0; i < len(y); i++ {
		SSres = SSres + math.Pow(y[i]-y_hat[i], 2)
		SStol = SStol + math.Pow(y[i]-yMean, 2)
	}

	return 1 - (SSres / SStol)
}

func Sum(v []float64) float64 {
	resp := 0.0
	for i := 0; i < len(v); i++ {
		resp += v[i]
	}

	return resp
}

func SumAbs(v []float64) float64 {
	resp := 0.0
	for i := 0; i < len(v); i++ {
		resp += math.Abs(v[i])
	}

	return resp
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

func RandomizeDataset(X, y mat.Matrix) (mat.Matrix, mat.Matrix) {
	n_rows, n_cols := X.Dims()

	rand.Seed(time.Now().UTC().UnixNano())
	X_randomized := mat.NewDense(n_rows, n_cols, nil)
	y_randomized := mat.NewDense(n_rows, 1, nil)

	for i, value := range rand.Perm(n_rows) {
		for j := 0; j < n_cols; j++ {
			X_randomized.Set(i, j, X.At(value, j))
		}
		y_randomized.Set(i, 0, y.At(value, 0))
	}

	return X_randomized, y_randomized
}
