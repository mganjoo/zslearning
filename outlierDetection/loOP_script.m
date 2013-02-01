addpath ../toolbox/pwmetric;

% use this many points to define the neighborhood 
kNN = 20;
lambda = 3;
load('ex8data1.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compute Local Outlier Probabilities  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for data points X (trained mapped images later)
% evaluate on Xval  (zero-shot test images later)

%compute all nearest neighbors
allDist = slmetric_pw(X',X','eucdist');

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

%  Visualize the example dataset
fprintf('Visualizing example dataset for outlier detection.\n\n');
hold on
plot(X(:, 1), X(:, 2), 'bx');
%axis([0 30 0 30]);

plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
disp(['Found ' num2str(sum(outliers)) ' outliers'])
hold off

% now run for new validation/test set Xval
% Xval = Xval(1:100,:);yval = yval(1:100);
allDist_test = slmetric_pw(X',Xval','eucdist');

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

fprintf('Visualizing test example dataset for outlier detection.\n\n');
hold on
plot(Xval(:, 1), Xval(:, 2), 'mx');
%axis([0 30 0 30]);

plot(Xval(outliers_test, 1), Xval(outliers_test, 2), 'go', 'LineWidth', 2, 'MarkerSize', 10);
disp(['Found ' num2str(sum(outliers_test)) ' test outliers, ' num2str(sum(outliers_test==yval')) '/' num2str(length(yval)) ' correct.'])
hold off