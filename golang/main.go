package main

import (
	"os"
)

func main() {
	switch argsWithProg := os.Args[len(os.Args)-1]; argsWithProg {
	case "linear":
		LinearRegression()

	case "polynomial":
		PolynomialRegression()

	case "simple":
		SimpleLinearRegression()

	case "rbf":
		RBFRegression()
	}
}
