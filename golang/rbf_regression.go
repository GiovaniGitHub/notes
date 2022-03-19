package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type RBFRegressionStruct struct {
	NumCenters int
	Centers    mat.Matrix
	Beta       float64
	Weight     mat.Matrix
}

func (rbf *RBFRegressionStruct) SetCenters(X mat.Matrix) {
	n_rows, n_cols := X.Dims()

	rand.Seed(time.Now().UTC().UnixNano())
	NewCenters := mat.NewDense(rbf.NumCenters, n_cols, nil)

	for i, value := range rand.Perm(n_rows)[:rbf.NumCenters] {
		for j := 0; j < n_cols; j++ {
			NewCenters.Set(i, j, X.At(value, j))
		}
	}

	rbf.Centers = NewCenters
}

func RadialBasisVector(center mat.Matrix, vector mat.Matrix, beta float64) float64 {
	var norm float64
	n_rows_center, n_cols_center := center.Dims()
	n_rows_vector, n_cols_vector := vector.Dims()

	if n_rows_center != n_rows_vector || n_cols_center != n_cols_vector {
		fmt.Errorf("Need same dimension")
	}
	for i := 0; i < n_rows_center; i++ {
		for j := 0; j < n_cols_center; j++ {
			norm += math.Pow(center.At(i, j)-vector.At(i, j), 2)
		}
	}
	return math.Exp(-beta * math.Sqrt(norm))
}

func (rbf *RBFRegressionStruct) CalculateGradient(X mat.Matrix) mat.Matrix {
	n_rows, _ := X.Dims()
	n_centers, _ := rbf.Centers.Dims()
	gradientArray := mat.NewDense(n_rows, n_centers, nil) //([]float64, n_rows*n_centers)
	for i := 0; i < n_rows; i++ {
		row_data := GetRowMatrix(X, i)
		for j := 0; j < n_centers; j++ {
			row_center := GetRowMatrix(rbf.Centers, j)
			gradientArray.Set(i, j, RadialBasisVector(row_data, row_center, rbf.Beta))
		}
	}

	return gradientArray
}

func (rbf *RBFRegressionStruct) Fit(X mat.Matrix, y mat.Matrix, Type TypeFactoration) {
	gradient := rbf.CalculateGradient(X)
	switch Type {
	case QR:
		qr := new(mat.QR)
		qr.Factorize(gradient)
		resp := new(mat.Dense)
		qr.SolveTo(resp, false, y)
		rbf.Weight = resp
	case SVD:
		svd := new(mat.SVD)
		svd.Factorize(gradient, mat.SVDFull)
		resp := new(mat.Dense)
		svd.SolveTo(resp, y, rbf.NumCenters)
		rbf.Weight = resp
	default:
		svd := new(mat.SVD)
		svd.Factorize(gradient, mat.SVDFull)
		resp := new(mat.Dense)
		svd.SolveTo(resp, y, rbf.NumCenters)
		rbf.Weight = resp
	}
}

func (rbf *RBFRegressionStruct) Predict(X mat.Matrix) []float64 {
	gradient := rbf.CalculateGradient(X)
	M := new(mat.Dense)
	M.Mul(rbf.Weight.T(), gradient.T())
	return M.RawMatrix().Data
}

func RBFRegression() {
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
		value_y, _ := strconv.ParseFloat(lines[i][1], 64)
		y_dense.Set(i-1, 0, value_y)
		for j := 0; j < n_cols; j++ {
			value_x, _ := strconv.ParseFloat(lines[i][0], 64)
			value_x = math.Pow(value_x, float64(j+1))
			X_dense.Set(i-1, j, value_x)
		}
	}

	RBF := RBFRegressionStruct{NumCenters: 20, Centers: nil, Beta: 4.0}
	RBF.SetCenters(X_dense)

	RBF.Fit(X_dense, y_dense, SVD)
	y_hat_svd := RBF.Predict(X_dense)

	RBF.Weight = nil

	RBF.Fit(X_dense, y_dense, QR)
	y_hat_qr := RBF.Predict(X_dense)

	idx := []float64{}
	for i := 0; i < len(y_hat_svd); i++ {
		idx = append(idx, float64(i*2))
	}

	p := plot.New()

	p.Title.Text = "RBF Regression"

	plotutil.AddScatters(p,
		"Original", GeneratePoints(idx, y_dense.RawMatrix().Data),
		fmt.Sprintf("SVD %.3f", r2(y_hat_svd, y_dense.RawMatrix().Data)), GeneratePoints(idx, y_hat_svd),
		fmt.Sprintf("QR %.3f", r2(y_hat_qr, y_dense.RawMatrix().Data)), GeneratePoints(idx, y_hat_qr),
	)

	if err := p.Save(7*vg.Inch, 7*vg.Inch, "rbf_regression_golang.png"); err != nil {
		panic(err)
	}

}
