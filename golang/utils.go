package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/montanaflynn/stats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot/plotter"
)

type TypeFactoration string

const (
	SVD TypeFactoration = "SVD"
	QR                  = "QR"
)

type Parameters struct {
	X             mat.Matrix
	y             mat.Matrix
	w             []float64
	b             float64
	epochs        int
	losses        []float64
	lr            float64
	func_type     string
	is_stochastic bool
	delta         float64
}

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

func SplitDataset(X mat.Matrix, y mat.Matrix, percent float64) (mat.Matrix, mat.Matrix, mat.Matrix, mat.Matrix) {
	X, y = RandomizeDataset(X, y)
	n_rows, n_cols := X.Dims()
	qtd_training := int(percent * float64(n_rows))

	y_train_dense := mat.NewDense(qtd_training, 1, nil)
	X_train_dense := mat.NewDense(qtd_training, n_cols, nil)

	y_test_dense := mat.NewDense(n_rows-qtd_training, 1, nil)
	X_test_dense := mat.NewDense(n_rows-qtd_training, n_cols, nil)

	for i := 0; i < n_rows; i++ {
		if i < qtd_training {
			for j := 0; j < n_cols; j++ {
				X_train_dense.Set(i, j, X.At(i, j))
			}
			y_train_dense.Set(i, 0, y.At(i, 0))
		} else {
			for j := 0; j < n_cols; j++ {
				X_test_dense.Set(i-qtd_training, j, X.At(i, j))
			}
			y_test_dense.Set(i-qtd_training, 0, y.At(i, 0))
		}
	}

	return X_train_dense, y_train_dense, X_test_dense, y_test_dense
}

func GetRowMatrix(x mat.Matrix, idx int) mat.Matrix {
	_, n_cols_data := x.Dims()
	row_data := make([]float64, n_cols_data)
	for ii := 0; ii < n_cols_data; ii++ {
		row_data[ii] = x.At(idx, ii)
	}
	return mat.NewDense(1, n_cols_data, row_data)
}

func GetRowData(x mat.Matrix, idx int) []float64 {
	_, n_cols_data := x.Dims()
	row_data := make([]float64, n_cols_data)
	for ii := 0; ii < n_cols_data; ii++ {
		row_data[ii] = x.At(idx, ii)
	}
	return row_data
}

func EuclideanDistance(v1 []float64, v2 []float64) float64 {
	if len(v1) != len(v2) {
		fmt.Errorf("Need same dimension")
	}
	acc := 0.0
	for ii := 0; ii < len(v1); ii++ {
		acc += math.Pow(v1[ii]-v2[ii], 2)
	}
	return math.Sqrt(acc)
}

func Cosine(a []float64, b []float64) float64 {
	count := 0
	length_a := len(a)
	length_b := len(b)
	if length_a > length_b {
		count = length_a
	} else {
		count = length_b
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		if k >= length_a {
			s2 += math.Pow(b[k], 2)
			continue
		}
		if k >= length_b {
			s1 += math.Pow(a[k], 2)
			continue
		}
		sumA += a[k] * b[k]
		s1 += math.Pow(a[k], 2)
		s2 += math.Pow(b[k], 2)
	}

	return (sumA / (math.Sqrt(s1) * math.Sqrt(s2)))
}

func Counter(v []int) map[int]int {
	freq := make(map[int]int)
	for _, num := range v {
		freq[num] = freq[num] + 1
	}

	return freq
}

func Unique(v []int) []int {
	keys := make(map[int]bool)
	list := []int{}
	for _, entry := range v {
		if _, value := keys[entry]; !value {
			keys[entry] = true
			list = append(list, entry)
		}
	}
	return list
}
