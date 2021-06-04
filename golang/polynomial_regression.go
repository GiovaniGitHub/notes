package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
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

func main() {
	csvfile, err := os.Open("../dataset/polynomial_regression_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	lines, err := csv.NewReader(csvfile).ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	n_rows := len(lines)
	n_cols := 7
	y_dense := mat.NewDense(n_rows-1, 1, nil)
	X_dense := mat.NewDense(n_rows-1, n_cols, nil)

	for i := 1; i < n_rows; i++ {
		value_x, _ := strconv.ParseFloat(lines[i][0], 64)
		value_y, _ := strconv.ParseFloat(lines[i][1], 64)
		y_dense.Set(i-1, 0, value_y)
		for j := 0; j < n_cols; j++ {
			value_x, _ = strconv.ParseFloat(lines[i][0], 64)
			value_x = math.Pow(value_x, float64(j+1))
			X_dense.Set(i-1, j, value_x)
		}
	}

	w := []float64{}
	w2 := []float64{}
	w3 := []float64{}

	for i := 0; i < n_cols; i++ {
		w = append(w, 0)
		w2 = append(w2, 0)
		w3 = append(w3, 0)
	}

	losses := []float64{}
	losses2 := []float64{}
	losses3 := []float64{}

	b := 0.0
	b2 := 0.0
	b3 := 0.0

	paramentersMSE := Parameters{X_dense, y_dense, w, b, 1000, losses, 0.01, "mse", false, 0}
	paramentersMAE := Parameters{X_dense, y_dense, w2, b2, 1000, losses2, 0.01, "mae", false, 0}
	paramentersHuber := Parameters{X_dense, y_dense, w3, b3, 1000, losses3, 0.01, "huber", false, 0.5}

	w, b = AdjustWeight(paramentersMSE)
	w2, b2 = AdjustWeight(paramentersMAE)
	w3, b3 = AdjustWeight(paramentersHuber)

	y_hat := Predict(w, X_dense, b)
	y_hat2 := Predict(w2, X_dense, b2)
	y_hat3 := Predict(w3, X_dense, b3)

	idx := []float64{}
	for i := 0; i < len(y_hat); i++ {
		idx = append(idx, float64(i*2))
	}

	p := plot.New()

	p.Title.Text = fmt.Sprint("Poly Regression \n R2 = ", r2(y_hat, y_dense.RawMatrix().Data))

	plotutil.AddLines(p,
		"Original", GeneratePoints(idx, y_dense.RawMatrix().Data),
		"Predicted MSE", GeneratePoints(idx, y_hat),
		"Predicted MAE", GeneratePoints(idx, y_hat2),
		"Predicted Huber", GeneratePoints(idx, y_hat3),
	)

	if err := p.Save(7*vg.Inch, 7*vg.Inch, "polynomial_regression_golang.png"); err != nil {
		panic(err)
	}
}
