# # Import Packages
# using Pkg  # Package to install new packages

# # Install packages 
# Pkg.add("DataFrames")
# Pkg.add("CSV")

# Load the installed packages
using DataFrames
using CSV
using Plots
using Random

function split_dataset(X::Vector{Float64}, y:: Vector{Float64}, percent::Float64)
    rng = MersenneTwister(1234);
    indexs = shuffle!(rng, Vector(1:length(y)))

    X_train = Vector{Float64}([])
    X_test = Vector{Float64}([])
    y_train = Vector{Float64}([])
    y_test = Vector{Float64}([])

    for i in 1:length(indexs)
        if i < percent*length(indexs)
            append!(X_train,Float64(X[indexs[i]]))
            append!(y_train,Float64(y[indexs[i]]))
        else
            append!(X_test,Float64(X[indexs[i]]))
            append!(y_test,Float64(y[indexs[i]]))            
        end

    end

    return X_train, y_train, X_test, y_test
end


function mse(y::Vector{Float64}, y_hat::Matrix{Float64})
    return sum((y_hat - y).^2)/length(y)
end


function expand_matrix(data:: Vector{Float64}, degree::Int)
    m_expanded = zeros(length(data),degree)
    for i = 1:size(m_expanded,1)
        for j = 1:size(m_expanded,2)
            m_expanded[i,j] = data[i]^(j-1)
        end          
    end

    return m_expanded
end


function gradient_descendent(X::Matrix{Float64}, y::Vector{Float64}, y_hat:: Matrix{Float64})
    n_rows = size(X,1)

    dw = (1/n_rows) * (transpose(X) *(y_hat - y))
    db = (1/n_rows) * sum(y_hat - y)

    return dw, db
end


function estimate_coef(x::Vector{Float64}, y::Vector{Float64}, degrees::Int, epochs::Int, learning_rate::Float64)
    X = expand_matrix(x, degrees)
    n_rows, n_cols = size(X)

    w = zeros(n_cols,1)
    b = 0

    losses = []
    for _ in 1:epochs
        y_hat = X*w + ones(n_rows)*b
        dw, db = gradient_descendent(X, y, y_hat)
        w -= learning_rate*dw
        b -= learning_rate*db
        
        error = mse(y, X * w + ones(n_rows)*b)
        append!(losses, error)
    end

    return w, b, losses

end

function estimate_coef_with_batch(x::Vector{Float64}, y::Vector{Float64}, bs::Int, degrees::Int, epochs::Int,
                                  learning_rate::Float64)
    X = expand_matrix(x, degrees)
    n_rows, n_cols = size(X)

    w = zeros(n_cols,1)
    b = 0

    losses = []
    for _ in 1:epochs
        for i in 1:bs:n_rows
            Xb = X[i:min(i + bs - 1, end),:]
            yb = y[i:min(i + bs - 1, end)]
            
            y_hat = Xb * w + ones(size(Xb,1)) * b

            dw, db = gradient_descendent(Xb, yb, y_hat)

            w -= learning_rate*dw
            b -= learning_rate*db
            
            error = mse(y, X * w + ones(size(X,1))*b)
            append!(losses, error)
        end
    end

    return w, b, losses

end

function main()
    PATH_FILE = "../dataset/polynomial_regression_data.csv"
    df = DataFrame(CSV.File(PATH_FILE))

    y = df.y
    X = df.x

    x_train, y_train, x_test, y_test = split_dataset(X, y, 0.7)

    w, b, losses = estimate_coef_with_batch(x_train, y_train, 10, 14, 2000, 0.01)

    X_test = expand_matrix(x_test, 14)
    y_hat = X_test * w + ones(size(X_test,1))*b
    yy = [y_test, y_hat]
    r = sum(losses)/length(losses)

    pp = plot(x_test, yy, title = "Polynomial Linear Regression: \n $r",seriestype = :scatter,
                label = ["Original" "Predicted"], lw = 2)
                
    savefig(pp, "polynomial_linear_regression.png")
end

main()