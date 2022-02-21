using LinearAlgebra: norm
using DataFrames
using CSV
using Statistics
using Plots
include("utils.jl")

mutable struct RBFRegression
    num_center::Integer
    centers::Array{Float64,2}
    beta::Float64
    w::Array{Float64,2}
    RBFRegression() = new()

end

function radial_basis_function(vector::Vector{Float64}, beta:: Float64)
    return exp( -beta * norm(vector)^2)
end

function calculate_gradient(X::Array{Float64,2}, self:: RBFRegression)
    gradient = zeros(size(self.centers,1), size(X,1))
    for i in 1:size(gradient,1)
        for j in 1:size(gradient,2)
            gradient[i,j] = radial_basis_function(self.centers[i,:] - X[j,:], self.beta)
        end
    end

    return transpose(gradient)
end

function fit(X::Array{Float64,2}, y::Array{Float64,2}, self::RBFRegression)
    rng = MersenneTwister(1234);
    idx = shuffle!(rng, Vector(1:self.num_center))
    self.centers = X[idx,:]
    gradient = calculate_gradient(X, self)
    self.w = gradient \ y
end

function predict(X:: Array{Float64,2}, self:: RBFRegression)
    gradient = calculate_gradient(X, self)
    return gradient*self.w
end


function main()
    PATH_FILE = "../dataset/polynomial_regression_data.csv"
    df = DataFrame(CSV.File(PATH_FILE))

    y = reshape(df.y,(length(df.y),1))
    x = reshape(df.x,(length(df.x),1))

    x_train, y_train, x_test, y_test = split_dataset(x, y, 0.7)
    X_train_expanded = expand_matrix(x_train, 7)
    x_test_expanded = expand_matrix(x_test, 7)

    rbf = RBFRegression()
    rbf.beta = 4
    rbf.num_center = 20

    fit(X_train_expanded,y_train,rbf)
    y_target = predict(x_test_expanded, rbf)

    m = mse(y_target, y_test)
    pp = plot(x_test, [y_test, y_target], title = " RBF Regression: \n $m",seriestype = :scatter,
                label = ["Original" "Predicted"], lw = 2)
                
    savefig(pp, "rbf_regression.png")
end
main()