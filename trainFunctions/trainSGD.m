function [ theta ] = trainSGD( trainParams, dataToUse, theta )

anneal = 0.00001 * [0.1 0.1 0.1 0.05 0.05 0.05 0.02 0.02 0.01 0.01 ...
      0.005 0.005 0.002 0.002 0.001 0.001 0.001 0.0005];

allCosts = zeros(1, length(anneal));
for iter=1:min(length(anneal), trainParams.maxIter)
   [ t, theta ] = trainParams.costFunction(theta, anneal(iter), dataToUse, trainParams);
   allCosts(iter) = t;
end

fprintf('Cost: %f\n', allCosts(1:iter));

end

