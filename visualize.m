function [t] = visualize(theta, trainParams, X, Y, wordTable, label_names)
    addpath toolbox/;
    addpath tsne/;
    [W, b] = stack2param(theta, trainParams.decodeInfo);
    mX = trainParams.f(bsxfun(@plus, W{1} * X, b{1}));
    t = tsne([mX wordTable]', [Y 1:10]');
    hold on;
    numImages = size(mX, 2);
    scatter(t(numImages+(1:10),1), t(numImage+(1:10),2), 100, 1:10, 'filled');
    for i = 1:10
        text(t(numImages+i,1),t(numImages+i,2),label_names{i});        
    end
    hold off;
end