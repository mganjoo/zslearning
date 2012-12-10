function [mappedX, mappedWordTable] = map2D(theta, trainParams, X, wordTable)
    addpath toolbox/;
    addpath tsne/;
    numImages = size(X, 2);
    [W, b] = stack2param(theta, trainParams.decodeInfo);
    mX = trainParams.f(bsxfun(@plus, W{1} * X, b{1}));
    t = tsne([mX wordTable]');
    mappedX = t(1:numImages, :);
    mappedWordTable = t(numImages+1:end, :);
end