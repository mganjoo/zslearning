% outputs the probability that each image feature is seen before
function [logprobability] = predictGaussianDiscriminant(projectedImageFeatures, mu, sigma_elem, priors, deletedClasses)

dim = size(mu, 1);
numLabels = size(mu, 1);
numTraining = size(projectedImageFeatures, 2);

probability = zeros(1, numTraining);
for i = 1:numLabels
    if ~ismember(i, deletedClasses)
        temp = bsxfun(@minus, projectedImageFeatures, mu(i,:)');
        logprobability = -0.5*(sum(1/sigma_elem(i)*(temp.^2), 1) + dim*log(2*pi) + dim*log(sigma_elem(i)));
        probability = probability + priors(i)*exp(logprobability);
    end
end

logprobability = log(probability);

end
