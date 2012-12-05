function [ theta, decodeInfo ] = mapInitParameters(trainParams)

if strcmp(func2str(trainParams.costFunction), 'mapOneShotNoAutoenc') == 0
    layers = [trainParams.inputSize trainParams.outputSize];
else
    layers = [trainParams.inputSize trainParams.outputSize trainParams.inputSize];
end

W = cell(length(layers)-1, 1);
b = cell(length(layers)-1, 1);

r = sqrt(6) / sqrt(sum(layers));
for i = 1:(length(layers)-1)
    W{i} = rand(layers(i+1), layers(i)) * 2*r-r;
    b{i} = zeros(layers(i+1), 1);
end

% Flatten
[theta, decodeInfo] = param2stack(W,b);

end

