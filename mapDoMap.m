function [ mapped ] = mapDoMap( images, theta, trainParams )

addpath toolbox/;

[ W, b ] = stack2param(theta, trainParams.decodeInfo);

% Feedforward
if (length(W) == 2)
    a2All = trainParams.f(bsxfun(@plus, W{1} * images, b{1}));
    mapped = bsxfun(@plus, W{2} * a2All, b{2});
else
    mapped = bsxfun(@plus, W{1} * images, b{1});
end

end