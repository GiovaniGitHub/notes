using DataFrames
using CSV
using Statistics
using Plots
include("utils.jl")

function sigmoid(z::Matrix{Float64})
    return 1 ./(1 .+ exp.(-z))
end

function estimate_coef(X::Matrix{Float64}, y::Matrix{Float64}, iterations::Int64, learning_rate::Float64)
    n_rows, n_cols = size(X)
    w = zeros((n_cols,1))
    b = 0
    
    loss = []
    
    for _ in 1:iterations
        z = X * w .+ b
        pred = sigmoid(z)
        
        cost = (-1/n_rows)*sum(transpose(y) * log.(pred) + (1 .- transpose(y)) * log.(1 .- pred))
        append!(loss, cost)
        
        dw = (1/n_rows) .* (transpose(X) * (pred .- y))
        db = (1/n_rows) * sum(pred .- y)
        
        w = w .- learning_rate .* dw
        b = b - learning_rate*db  
    end

    return w, b, loss
end

function main()
    PATH_FILE = "../dataset/logistic_regression.csv"
    df = DataFrame(CSV.File(PATH_FILE))

    y = float(reshape(df.y,(length(df.y),1)))
    X = Matrix(select!(df, Not(:y)))

    x_train, y_train, x_test, y_test = split_dataset(X, y, 0.9)

    w, b, losses = estimate_coef(x_train, y_train, 2000, 0.1)
    r = sum(losses)/length(losses)
    y_hat = x_test * w .+ b
    y_hat = Matrix{Float64}(y_hat .> 0.5)

    yy = [y_test, y_hat]
    acc= sum(y_test .== y_hat)/length(y_test)
    
    pp = plot( yy, title = "Logistic Linear Regression: \n Accuracy $acc",seriestype = :scatter, label = ["Original" "Predicted"], lw = 2)
                
    savefig(pp, "logistic_regression.png")
end

main()