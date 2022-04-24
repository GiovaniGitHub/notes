using DataFrames
using CSV
# using Distances
using StatsBase:mode
using Formatting

include("utils.jl")
include("distances.jl")
mutable struct KNN
    data::Array{Float64,2}
    target::Array{Float64,2}
end

function predict(x::Array{Float64,1}, n_neighborhood:: Int64, self::KNN, distance::Function)
    heat_labels = zeros((n_neighborhood,2))
    heat_labels[:,1] .= Inf
    for i in 1:size(self.data,1)
        d = distance(x, self.data[i,:])
        idx = findfirst(>(d), heat_labels[:,1])
        if idx !== nothing
            heat_labels[idx, 1] = d
            heat_labels[idx, 2] = self.target[i]
        end
    end
    return mode(heat_labels[:,2])
end

function main()
    PATH_FILE = "../dataset/knn_classification.csv"
    df = DataFrame(CSV.File(PATH_FILE))
    select!(df, Not(:id))
    y = df.class
    y = reshape(y,(length(y),1))
    select!(df, Not(:class))

    X = Matrix{Float64}(df)
    y = Matrix{Float64}(y)
    x_train, y_train, x_test, y_test = split_dataset(X, y, 0.7)

    knn = KNN(x_train, y_train)

    n_neighborhood = 10
    for distance in all_functions
        println(distance)
        acc = sum([
                    y_test[i,1] === predict(x_test[i,:], n_neighborhood, knn, distance)[1,1]
                    for i in 1:size(x_test,1)
                ]) / size(y_test,1)
        printfmtln("With function {:s} the accuracy is {:f}", string(distance), acc)
        
    end
end

main()
