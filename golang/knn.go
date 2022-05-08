package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"sort"

	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type KNNClassifierStruct struct {
	X mat.Matrix
	y mat.Matrix
}

type NeighborhoodItem struct {
	distance float64
	class    string
}

func (knn *KNNClassifierStruct) predict(row []float64, nNeighborhood int) string {
	neighborhood := []NeighborhoodItem{}

	//Init neighborhood
	for i := 0; i < nNeighborhood; i++ {
		neighborhood = append(neighborhood, NeighborhoodItem{distance: math.MaxFloat64, class: ""})
	}

	n_rows, n_cols := knn.X.Dims()
	if n_cols != len(row) {
		errors.New("Zero cannot be used")
	}
	for i := 0; i < n_rows; i++ {
		row_dense := GetRowData(knn.X, i)
		distance := EuclideanDistance(row_dense, row)
		sort.SliceStable(neighborhood, func(i, j int) bool {
			return neighborhood[i].distance < neighborhood[j].distance
		})

		if neighborhood[nNeighborhood-1].distance > distance {
			neighborhood[nNeighborhood-1].distance = distance
			neighborhood[nNeighborhood-1].class = strconv.FormatInt(
				int64(
					math.Round(
						GetRowData(knn.y, i)[0])), 10)
		}
	}
	freq := make(map[string]int)
	for i := 0; i < nNeighborhood; i++ {
		freq[neighborhood[i].class] = freq[neighborhood[i].class] + 1
	}

	max_frequency := 0
	max_class := ""
	for class, frequency := range freq {
		if max_frequency < frequency {
			max_class = class
			max_frequency = frequency
		}
	}

	return max_class
}

func KNNClassifier() {
	csvfile, err := os.Open("../dataset/knn_classification.csv")
	if err != nil {
		log.Fatal(err)
	}
	lines, err := csv.NewReader(csvfile).ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	n_rows := len(lines)
	n_cols := len(lines[1]) - 2

	y_dense := mat.NewDense(n_rows-1, 1, nil)
	X_dense := mat.NewDense(n_rows-1, n_cols, nil)

	for i := 1; i < n_rows; i++ {
		value_y, _ := strconv.ParseFloat(lines[i][len(lines[i])-1], 64)
		y_dense.Set(i-1, 0, value_y)
		for j := 0; j < n_cols; j++ {
			value_x, _ := strconv.ParseFloat(lines[i][j+1], 64)
			X_dense.Set(i-1, j, value_x)
		}
	}
	x_train, y_train, x_test, y_test := SplitDataset(X_dense, y_dense, 0.7)
	n_rows_test, _ := y_test.Dims()

	knn := KNNClassifierStruct{x_train, y_train}
	acc := 0.0
	for i := 0; i < n_rows_test; i++ {
		sample := GetRowData(x_test, i)
		var y_hat = knn.predict(sample, 10)
		var target = strconv.FormatInt(
			int64(
				math.Round(
					GetRowData(y_test, i)[0])), 10)
		if y_hat == target {
			acc += 1.0
		}
	}
	fmt.Printf("Accuracy %.2f \n", acc/float64(n_rows_test))
}
