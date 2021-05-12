# # Import Packages
# using Pkg  # Package to install new packages

# # Install packages 
# Pkg.add("DataFrames")
# Pkg.add("CSV")

# Load the installed packages
using DataFrames
using CSV
using Statistics

function estimate_coef(x:: Vector{Float64}, y :: Vector{Float64})
    n = length(x)

    m_x = mean(x)
    m_y = mean(y)

    SS_xy = sum(y.*x) - n*m_y*m_x
    SS_xx = sum(x.*x) - n*m_x*m_x
    SS_yy = sum(y.*y) - n*m_y*m_y

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    r = SS_xy/(sqrt(SS_xx*SS_yy))
    return (b_0, b_1, r)
end


function main()
    PATH_FILE = "../dataset/simple_regression.csv"
    df = DataFrame(CSV.File(PATH_FILE))
    n_rows, n_cols = size(df)
    
    y = df.happiness
    x = df.income

    b_0, b_1, r= estimate_coef(x,y)
    println("b_0: ", b_0)
    println("b_1: ", b_1)
    print("residual: ", r)
end

main()