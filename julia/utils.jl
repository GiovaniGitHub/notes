using Random


function split_dataset(X::Vector{Float64}, y:: Vector{Float64}, percent::Float64)
    rng = MersenneTwister(1234);
    indexs = shuffle!(rng, Vector(1:length(y)))

    X_train = Vector{Float64}([])
    X_test = Vector{Float64}([])
    y_train = Vector{Float64}([])
    y_test = Vector{Float64}([])

    for i in 1:length(indexs)
        if i <= percent*length(indexs)
            append!(X_train,Float64(X[indexs[i]]))
            append!(y_train,Float64(y[indexs[i]]))
        else
            append!(X_test,Float64(X[indexs[i]]))
            append!(y_test,Float64(y[indexs[i]]))            
        end

    end

    return X_train, y_train, X_test, y_test
end


function gradient_descendent(X::Matrix{Float64}, y::Vector{Float64}, y_hat:: Matrix{Float64})
    n_rows = size(X,1)

    dw = (1/n_rows) * (transpose(X) *(y_hat - y))
    db = (1/n_rows) * sum(y_hat - y)

    return dw, db
end


function mse(y::Vector{Float64}, y_hat::Matrix{Float64})
    return sum((y_hat - y).^2)/length(y)
end