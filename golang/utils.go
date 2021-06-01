package main

import (
	"math"

	"github.com/montanaflynn/stats"
	"gonum.org/v1/plot/plotter"
)

func generatePoints(x []float64, y []float64) plotter.XYs {
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
