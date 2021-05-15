# # Import Packages
# using Pkg  # Package to install new packages

# # Install packages 
# Pkg.add("DataFrames")
# Pkg.add("CSV")

# Load the installed packages
using DataFrames
using CSV
using Plots


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
    b = zeros(n_rows,1)

    losses = []
    for _ in 1:epochs
        y_hat = X * w + b
        dw, db = gradient_descendent(X, y, y_hat)
        w -= learning_rate*dw
        b -= learning_rate*ones(length(y))*db
        
        error = mse(y, X * w + b)
        append!(losses, error)
    end

    return w, b, losses

end


function main()
    PATH_FILE = "../dataset/polynomial_regression_data.csv"
    df = DataFrame(CSV.File(PATH_FILE))

    y = df.y
    x = df.x

    w, b, losses = estimate_coef(x, y, 12, 5000, 0.01)

    X = expand_matrix(x, 12)
    y_hat = X * w + b
    yy = [y,y_hat]
    r = sum(losses)/length(losses)

    pp = plot(x, yy, title = "Polynomial Linear Regression: \n $r",seriestype = :scatter,
                label = ["Original" "Predicted"], lw = 2)
                
    savefig(pp, "polynomial_linear_regression.png")
end

main()