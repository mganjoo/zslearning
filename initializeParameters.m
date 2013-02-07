function [ theta, decodeInfo ] = initializeParameters(trainParams)

makeBias = true;
if strcmp(func2str(trainParams.costFunction), 'mapTrainingCost') 
    layers = [trainParams.inputSize trainParams.hiddenSize trainParams.outputSize];
elseif strcmp(func2str(trainParams.costFunction), 'cwTrainingCost')
    layers = [trainParams.embeddingSize + trainParams.imageColumnSize, trainParams.hiddenSize, 1];
elseif strcmp(func2str(trainParams.costFunction), 'softmaxCost')
    layers = [trainParams.inputSize trainParams.outputSize];
    makeBias = false;
else
    layers = [trainParams.inputSize trainParams.outputSize];
end

W = cell(length(layers)-1, 1);
if makeBias
    b = cell(length(layers)-1, 1);
end

r = sqrt(6) / sqrt(sum(layers));
for i = 1:(length(layers)-1)
    W{i} = rand(layers(i+1), layers(i)) * 2*r-r;
    if makeBias
        b{i} = zeros(layers(i+1), 1);
    end
end

% Flatten
if makeBias
    [theta, decodeInfo] = param2stack(W,b);
else
    [theta, decodeInfo] = param2stack(W);
end

end

