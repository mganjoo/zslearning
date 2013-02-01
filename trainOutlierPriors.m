function [ nplof, pdist ] = trainOutlierPriors(Xtrain, kNN, lambda)

addpath toolbox/pwmetric;

% for data points X (trained mapped images later)
% evaluate on Xtest  (zero-shot test images later)

% compute all nearest neighbors
allDist = slmetric_pw(Xtrain,Xtrain,'eucdist');

%first row are just the points, then follow the nearest neighbors
[allPointsNNDistances, allPointsNN] = sort(allDist);

S = allPointsNN(2:kNN+1,:);
sigma = sqrt(sum(allPointsNNDistances.^2)./kNN);
pdist = lambda * sigma;
Epdist = mean(pdist(S));
plof = pdist./Epdist -1;
nplof = lambda * sqrt(mean(plof.^2));

end