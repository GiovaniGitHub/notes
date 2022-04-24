using LinearAlgebra: norm, dot
using Statistics: mean, cov

function euclidean(x, y)
    return sqrt(sum(x - y) .^2)
end

function sqeuclidean(x, y)
    return sum((x - y).^2)
end

function cityblock(x, y)
    return sum(abs.(x - y))
end

function total_variation(x, y)
    return sum(abs.(x - y)) / 2
end

function jaccard(x, y)
    return 1 - sum(min(x, y)) / sum(max(x, y))
end

function cosine_distance(x, y)
    return 1 - dot(x, y) / (norm(x) * norm(y))
end

function correlation_distance(x, y)
    return cosine_distance(x .- mean(x), y .- mean(y))
end

function chi_square_distance(x, y)
    return sum((x - y).^2 / (x + y))
end

all_functions = [
        euclidean,
        sqeuclidean,
        cityblock,
        total_variation,
        jaccard,
        cosine_distance,
        correlation_distance,
        chi_square_distance,
        ];