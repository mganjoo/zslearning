function [t] = visualize(theta, trainParams, X, wordTable)
    addpath toolbox/;
    addpath tsne/;
    [W, b] = stack2param(theta, trainParams.decodeInfo);
    mX = trainParams.f(bsxfun(@plus, W{1} * X, b{1}));
    t = tsne([mX wordTable]');
end