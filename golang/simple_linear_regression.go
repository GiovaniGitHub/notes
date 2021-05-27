package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/montanaflynn/stats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func coef_estimate(x []float64, y []float64) (float64, float64, float64) {
	var x_mean, _ = stats.Mean(x)
	var y_mean, _ = stats.Mean(y)

	x_vec := mat.NewVecDense(len(x), x)
	y_vec := mat.NewVecDense(len(y), y)
	n := float64(len(y))

	SS_xy := mat.Dot(x_vec, y_vec) - n*y_mean*x_mean
	SS_xx := mat.Dot(x_vec, x_vec) - n*x_mean*x_mean
	SS_yy := mat.Dot(y_vec, y_vec) - n*y_mean*y_mean

	b_1 := SS_xy / SS_xx
	b_0 := y_mean - b_1*x_mean

	r := SS_xy / (math.Sqrt(SS_xx * SS_yy))

	return b_0, b_1, r

}

func main() {
	csvfile, err := os.Open("../dataset/simple_regression.csv")
	if err != nil {
		log.Fatal(err)
	}
	df := dataframe.ReadCSV(csvfile)
	income := df.Col("income").Float()
	happiness := df.Col("happiness").Float()

	b_0, b_1, r := coef_estimate(income, happiness)

	y_hat := make([]float64, len(income))
	error := make([]float64, len(income))
	for i, _ := range income {
		y_hat[i] = income[i]*b_1 + b_0
		error[i] = math.Abs(y_hat[i] - happiness[i])
	}

	p := plot.New()

	p.Title.Text = fmt.Sprint("Simple Linear Regression \t r2 = ", r)

	plotutil.AddScatters(p,
		"Original", generatePoints(income, happiness),
		"Predicted", generatePoints(income, y_hat))

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "points.png"); err != nil {
		panic(err)
	}
}
