# # Import Packages
# using Pkg  # Package to install new packages

# # Install packages 
# Pkg.add("DataFrames")
# Pkg.add("CSV")

# Load the installed packages
using DataFrames
using CSV
using Statistics
using Plots

function estimate_coef_matrix(X:: Matrix{Float64}, y::Vector{Float64})
    n_rows,_ = size(X)
    X = hcat(X, ones(n_rows))

    theta = inv(transpose(X) * X) * transpose(X) * y

    return theta
end


function main()
    PATH_FILE = "../dataset/linear_regression2.csv"
    df = DataFrame(CSV.File(PATH_FILE))
    
    y = df.z
    X = Matrix(select!(df, Not(:z)))

    theta = estimate_coef_matrix(X,y)

    n_rows,_ = size(X)
    X = hcat(X, ones(n_rows))

    y_hat = X * theta

    yy = [y,y_hat]

    r = 1 - (sum(y) - n_rows*mean(y)) / mean((y .- y_hat).^2)

    println(r)
end

main()