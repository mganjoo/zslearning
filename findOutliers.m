addpath toolbox/pwmetric;

% use this many points to define the neighborhood 
kNN = 20;
lambda = 3;
load('mappedTrainData.mat');
X = mappedX;
load('mappedTestData.mat');
Xval = mappedX;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compute Local Outlier Probabilities  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for data points X (trained mapped images later)
% evaluate on Xval  (zero-shot test images later)

%compute all nearest neighbors
allDist = slmetric_pw(X,X,'eucdist');

%first row are just the points, then follow the nearest neighbors
[allPointsNNDistances, allPointsNN] = sort(allDist);

S = allPointsNN(2:kNN+1,:);
Sdist = allPointsNNDistances(2:kNN+1,:);
sigma = sqrt(sum(allPointsNNDistances.^2)./kNN);
pdist = lambda * sigma;
Epdist = mean(pdist(S));
plof = pdist./Epdist -1;
nplof = lambda * sqrt(mean(plof.^2));
loop = erf(plof./(nplof * sqrt(2)));
loop(loop<0) = 0;

% outlier threshold 
outliers = loop>=0.5;

% now run for new validation/test set Xval
allDist_test = slmetric_pw(X,Xval,'eucdist');

% first row are just the points, then follow the nearest neighbors
% for each test point, we look at the closest training points only
[allPointsNNDistances_test, allPointsNN_test] = sort(allDist_test);
S_test = allPointsNN_test(2:kNN+1,:);
Sdist_test = allPointsNNDistances_test(2:kNN+1,:);
sigma_test = sqrt(sum(allPointsNNDistances_test.^2)./kNN);
pdist_test = lambda * sigma_test;
Epdist_test = mean(pdist(S_test)); 
plof_test = pdist_test./Epdist_test -1;
loop_test = erf(plof_test./(nplof * sqrt(2))); % use nplof from train
loop_test(loop_test<0) = 0;
outliers_test = loop_test>=0.5;
