package main

import "gonum.org/v1/plot/plotter"

func generatePoints(x []float64, y []float64) plotter.XYs {
	pts := make(plotter.XYs, len(y))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}
	return pts
}
