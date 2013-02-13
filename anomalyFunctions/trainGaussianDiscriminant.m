%p(y|x,X_T) = sum_v p(y|v,X_t)*p(v|x,X_T)
%p(y=1|v,x,X_T) = p(x|y=i,mu,Sigma)*p(y=i)/sum_{k=1}^{|V|}p(x|y=k,mu,Sigma)*pi(y=k)

function [mu,sigma_elem,priors] = trainGaussianDiscriminant(projectedImageFeatures, labels, numLabels, wordVectors)

[dim,numTraining] = size(projectedImageFeatures);
%sigma = zeros(numLabels, dim, dim);
%mu = zeros(numLabels, dim);
sigma_elem = zeros(numLabels, 1);
mu = wordVectors';

priors = zeros(numLabels, 1);

for i=1:numLabels
	labelImageFeatures = projectedImageFeatures(:, labels == i);
    priors(i) = size(labelImageFeatures, 2) / numTraining;
    labelMu = squeeze(mu(i,:))';
    sigma_elem(i) = sum(sum(bsxfun(@minus, labelImageFeatures, labelMu).^2))/(numTraining*dim);
%    sigma(i,:,:) = diag(repmat(sum(sum(bsxfun(@minus, labelImageFeatures, labelMu).^2))/(numTraining*dim), dim, 1));
end

end
