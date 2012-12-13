function [ theta ] = trainLBFGS( trainParams, dataToUse, theta )

% minFunc options
options.Method = 'lbfgs';
options.display = 'on';
options.MaxIter = trainParams.maxIter;
[theta, ~, ~, ~] = minFunc( @(p) trainParams.costFunction(p, dataToUse, trainParams ), theta, options);

end

