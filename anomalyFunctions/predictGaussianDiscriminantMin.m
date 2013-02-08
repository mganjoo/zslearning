% outputs the probability that each image feature 
function [logprobability] = predictGaussianDiscriminantMin(projectedImageFeatures, mu, sigma, priors, deletedClasses)

dim = size(mu, 1);
numLabels = size(mu, 1);
numTraining = size(projectedImageFeatures, 2);

probability = zeros(numLabels - length(deletedClasses), numTraining);
k = 1;
for i = 1:numLabels
    if sum(deletedClasses == i) == 0
        labelSigmaInv = pinv(squeeze(sigma(i,:,:)));
        labelSigmaDet = det(squeeze(sigma(i,:,:)));
        labelMu = mu(i,:)';

        temp = projectedImageFeatures - repmat(labelMu,1,numTraining);
        logprobability = - 0.5*sum(temp.*(labelSigmaInv*temp), 1) - 0.5*dim*log(2*pi) - 0.5*log(labelSigmaDet);
        probability(k, :) = logprobability;
        k = k + 1;
    end
end

logprobability = max(probability);

end
