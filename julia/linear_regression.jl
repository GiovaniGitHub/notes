# # Import Packages
# using Pkg  # Package to install new packages

# # Install packages 
# Pkg.add("DataFrames")
# Pkg.add("CSV")

# Load the installed packages
using DataFrames
using CSV
using Statistics


function estimate_coef_matrix(X:: Matrix{Float64}, y::Vector{Float64})
    n_rows,n_columns = size(X)
    X = hcat(X, ones(n_rows))

    beta = inv(transpose(X) * X) * transpose(X) * y

    return beta[1:length(beta)-1], beta[length(beta)]
end


function main()
    PATH_FILE = "../dataset/linear_regression.csv"
    df = DataFrame(CSV.File(PATH_FILE))
    n_rows, n_cols = size(df)
    
    y = df.MEDV
    X = Matrix(select!(df, Not(:MEDV)))

    b_0, b_1= estimate_coef_matrix(X,y)

end

main()