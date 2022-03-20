# # Import Packages
# using Pkg  # Package to install new packages

# # Install packages 
# Pkg.add("DataFrames")
# Pkg.add("CSV")

# Load the installed packages
using DataFrames
using CSV
using Plots
using Statistics
include("utils.jl")
include("gradients.jl")
function estimate_coef_elastic_net(x::Vector{Float64}, y::Vector{Float64}, degrees::Int, epochs::Int, learning_rate::Float64, ridge_coef::Float64, lasso_coef::Float64)
    X = expand_matrix(x, degrees)
    w = zeros(n_cols,1)
    skip = False
    losses = []
    count_iter = 0
    while not skip
        count_iter+=1
        y_hat = X*w
        dif = (transpose(X) * (y_hat - y))

        dw = learning_rate*(dif + ridge_coef*sign.(w) + lasso_coef*2*w)
        w = w - dw
        mse_value = mse(y_hat, y)
        
        append!(losses, mse_value)
        if Statistics.mean(abs.(dw)) <= tol
            skip = True
        if count_iter==max_iterators
            skip = True

function estimate_coef(x::Vector{Float64}, y::Vector{Float64}, degrees::Int, epochs::Int, learning_rate::Float64, 
    func_adjust::Function)
    X = expand_matrix(x, degrees)
    n_rows, n_cols = size(X)

    w = zeros(n_cols,1)
    b = 0

    losses = []
    for _ in 1:epochs
        y_hat = X*w .+ b
        dw, db = func_adjust(X, y, y_hat)
        w -= learning_rate*dw
        b -= learning_rate*db
        
        error = mse(y, X * w .+ b)
        append!(losses, error)
    end

    return w, b, losses

end

function estimate_coef_with_batch(x::Matrix{Float64}, y::Matrix{Float64}, bs::Int, degrees::Int, epochs::Int,
                                  learning_rate::Float64, func_adjust::Function)
    X = expand_matrix(x, degrees)
    n_rows, n_cols = size(X)

    w = zeros(n_cols,1)
    b = 0

    losses = []
    for _ in 1:epochs
        for i in 1:bs:n_rows
            Xb = X[i:min(i + bs - 1, end),:]
            yb = y[i:min(i + bs - 1, end)]
            
            y_hat = Xb * w .+ b

            dw, db = func_adjust(Xb, yb, y_hat)

            w -= learning_rate*dw
            b -= learning_rate*db
            
            error = mse(y, X * w .+ b)
            append!(losses, error)
        end
    end

    return w, b, losses

end

function main()
    PATH_FILE = "../dataset/polynomial_regression_data.csv"
    df = DataFrame(CSV.File(PATH_FILE))

    y = df.y
    y = reshape(y,(length(y),1))
    X = df.x
    X = reshape(X,(length(X),1))

    x_train, y_train, x_test, y_test = split_dataset(X, y, 0.7)

    w, b, losses = estimate_coef_with_batch(x_train, y_train, 10, 14, 2000, 0.01, update_weights_mae)

    X_test = expand_matrix(x_test, 14)
    y_hat = X_test * w .+ b
    yy = [y_test, y_hat]
    r = sum(losses)/length(losses)

    pp = plot(x_test, yy, title = "Polynomial Linear Regression: \n $r",seriestype = :scatter,
                label = ["Original" "Predicted"], lw = 2)
                
    savefig(pp, "polynomial_linear_regression.png")
end

main()