function [ theta, decodeInfo ] = mapInitParameters(trainParams)

W = cell(2, 1);
b = cell(2, 1);

layers = [trainParams.inputSize trainParams.outputSize trainParams.inputSize];
r = sqrt(6) / sqrt(sum(layers));
for i = 1:(length(layers)-1)
    W{i} = rand(layers(i+1), layers(i)) * 2*r-r;
    b{i} = zeros(layers(i+1), 1);
end

% Flatten
[theta, decodeInfo] = param2stack(W,b);

end

