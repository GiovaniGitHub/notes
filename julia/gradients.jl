
function update_weights_mse(X::Matrix{Float64}, y::Vector{Float64}, y_hat:: Matrix{Float64})
    n_rows = size(X,1)

    dw = (1/n_rows) * (transpose(X) *(y_hat - y))
    db = (1/n_rows) * sum(y_hat - y)

    return dw, db
end

function update_weights_mae(X::Matrix{Float64}, y::Vector{Float64}, y_hat:: Matrix{Float64})

    dif = (transpose(X) * (y_hat - y))
    dw = (1/sum(abs.(y_hat - y))) * (dif)
    db = (1/sum(abs.(y_hat - y))) * sum(y_hat - y)

    return dw, db
end

function update_weights_huber(X::Matrix{Float64}, y::Vector{Float64}, y_hat:: Matrix{Float64}, delta:: Float64)
    n_rows = size(X,1)
    dif = (transpose(X) * (y_hat - y))

    if (sum(abs.(y - y_hat))) <= delta
        dw = (1/n_rows) * (transpose(X) *(y_hat - y))
        db = (1/n_rows) * sum(y_hat - y)
    else
        dw = (1/sum(abs.(y_hat - y))) * (dif)
        db = (1/sum(abs.(y_hat - y))) * sum(y_hat - y)
    end

    return dw, db
end

