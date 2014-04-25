% outputs the probability that each image feature 
function [logprobability] = predictGaussianDiscriminantMin(projectedImageFeatures, mu, sigma_elem, deletedClasses)

numLabels = size(mu, 1);
numTraining = size(projectedImageFeatures, 2);
dim = size(projectedImageFeatures, 1);

probability = zeros(numLabels - length(deletedClasses), numTraining);
k = 1;
for i = 1:numLabels
    if ~ismember(i, deletedClasses)
        %labelSigmaInv = pinv(squeeze(sigma(i,:,:)));
        %labelSigmaDet = det(squeeze(sigma(i,:,:)));

        temp = bsxfun(@minus, projectedImageFeatures, mu(i,:)');
        %logprobability = - 0.5*sum(temp.*(labelSigmaInv*temp), 1) - 0.5*log(labelSigmaDet);
        logprobability = -0.5*(sum(1/sigma_elem(i)*(temp.^2), 1) + dim*log(sigma_elem(i)));
        %probability(k, :) = logprobability;
        probability(k, :) = logprobability;
        k = k + 1;
    end
end

logprobability = max(probability);

end
