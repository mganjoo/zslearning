function sigmoid_prime = sigmoid_prime(a)
    sigmoid_prime = a .* (1 - a);
end
