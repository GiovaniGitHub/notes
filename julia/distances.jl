using LinearAlgebra: norm
using Distances:mahalanobis

function euclidean(x, y)
    return sqrt(sum(x - y) .^2)
end

function sqeuclidean(x, y)
    return sum((x - y).^2)
end

function cityblock(x, y)
    return sum(abs(x - y))
end

function total_variation(x, y)
    return sum(abs(x - y)) / 2
end

function jaccard(x, y)
    return 1 - sum(min(x, y)) / sum(max(x, y))
end

function cosine_distance(x, y)
    return 1 - dot(x, y) / (norm(x) * norm(y))
end

function correlation_distance(x, y)
    return cosine_distance(x - mean(x), y - mean(y))
end

function chi_square_distance(x, y)
    return sum((x - y).^2 / (x + y))
end

function kullback_leibler_divergence(x, y)
    return sum(x .* log(x ./ y))
end

function generalized_kullback_leibler_divergence(x, y)
    return sum(x .* log(x ./ y) - x + y)
end

function dice_distance(x, y)
    return 2*(sum(min(x, y)))/(norm(x) + norm(y))
end

function jensen_shannon_divergence(x, y)
    m = (x + y) / 2
    return kullback_leibler_divergence(x, m) / 2 + kullback_leibler_divergence(m, y) / 2
end

function mahalanobis(x, y)
    return sqrt(sum((x - y).^2)*cov(x, y))
end