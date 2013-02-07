function [ mapped ] = mapDoMap( images, theta, trainParams )

[ W, b ] = stack2param(theta, trainParams.decodeInfo);

% Feedforward
if strcmp(func2str(trainParams.costFunction), 'mapTrainingCostOneLayer')
    mapped = bsxfun(@plus, W{1} * images, b{1});
else
    a2All = trainParams.f(bsxfun(@plus, W{1} * images, b{1}));
    mapped = bsxfun(@plus, W{2} * a2All, b{2});
end

end