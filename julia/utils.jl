using Random


function split_dataset(X::Matrix{Float64}, y:: Matrix{Float64}, percent::Float64)
    rng = MersenneTwister(1234);
    x_n_rows, x_n_cols = size(X)
    y_n_rows, y_n_cols = size(y)
    indexs = shuffle!(rng, Vector(1:x_n_rows))

    X_train = Matrix{Float64}(undef,0,x_n_cols)
    X_test = Matrix{Float64}(undef,0,x_n_cols)
    y_train = Matrix{Float64}(undef,0,y_n_cols)
    y_test = Matrix{Float64}(undef,0,y_n_cols)

    for i in 1:length(indexs)
        if i <= percent*length(indexs)
            X_train = vcat(X_train, X[indexs[i],:]')
            y_train = vcat(y_train, y[indexs[i],:]')
        else
            X_test = vcat(X_test, X[indexs[i],:]')
            y_test = vcat(y_test, y[indexs[i],:]')    
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


function mse(y::Matrix{Float64}, y_hat::Matrix{Float64})
    return sum((y_hat .- y).^2)/length(y)
end