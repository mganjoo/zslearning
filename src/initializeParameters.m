function [theta, decodeInfo] = initializeParameters(trainParams)

% bottom layer
if isfield(trainParams, 'inputSize') == true
    % allow manual override
    inputSize = trainParams.inputSize;
else
    inputSize = trainParams.embeddingSize + trainParams.imageColumnSize;
end

% deep layers
W = cell(length(trainParams.layers)+1, 1);
b = cell(length(trainParams.layers), 1); % no bias at last layer

layers = [inputSize trainParams.layers 1];
r = sqrt(6) / sqrt(sum(layers));
for i = 1:(length(layers)-1)
    W{i} = rand(layers(i+1), layers(i)) * 2*r-r;
    if i ~= length(layers)-1
        b{i} = zeros(layers(i+1), 1);
    end
end

[theta, decodeInfo] = param2stack(W,b);
