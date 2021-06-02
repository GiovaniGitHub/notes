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
	n_cols := 8
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
	for i := 0; i < n_cols; i++ {
		w = append(w, 0)
	}
	losses := []float64{}
	b := 0.0
	w, b = AdjustWeight(X_dense, y_dense, w, b, 120, losses, 0.01, UpdateWeightsMSE, false)

	y_hat := Predict(w, X_dense, b)

	idx := []float64{}
	for i := 0; i < len(y_hat); i++ {
		idx = append(idx, float64(i*2))
	}

	p := plot.New()

	p.Title.Text = fmt.Sprint("Poly Regression \n R2 = ", r2(y_hat, y_dense.RawMatrix().Data))

	plotutil.AddLines(p,
		"Original", generatePoints(idx, y_dense.RawMatrix().Data),
		"Predicted", generatePoints(idx, y_hat),
	)

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "polynomial_regression_golang.png"); err != nil {
		panic(err)
	}
}
