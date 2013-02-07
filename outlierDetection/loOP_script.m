addpath ../toolbox/pwmetric;

% use this many points to define the neighborhood 
kNN = 20;
lambda = 3;
load('tsneMappedTrainDataSmall.mat');
load('tsneMappedTestDataSmall.mat');

numCategories = 10;
zeroCategories = [ 4, 10 ];
nonZeroCategories = setdiff(1:numCategories, zeroCategories);
numPerCategory = 700;

figure;
hold on
nplofAll = zeros(length(nonZeroCategories), 1);
pdistAll = zeros(length(nonZeroCategories), numPerCategory);
for categoryInd = 1:length(nonZeroCategories)
    filtered = find(trainY == nonZeroCategories(categoryInd));
    X = mappedTrainX(filtered(randi(length(filtered), 1, numPerCategory)), :);
    
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
    
    pdistAll(categoryInd, :) = pdist;
    nplofAll(categoryInd, :) = nplof;

    % outlier threshold 
    outliers = loop>=0.5;

    %  Visualize the example dataset
    plot(X(:, 1), X(:, 2), 'bx');

    plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
    disp(['Found ' num2str(sum(outliers)) ' outliers'])    
end
hold off

numImages = length(testY);
probs = zeros(length(nonZeroCategories), numImages);
for categoryInd = 1:length(nonZeroCategories)
    filtered = find(trainY == nonZeroCategories(categoryInd));
    X = mappedTrainX(filtered(randi(length(filtered), 1, numPerCategory)), :);
    
    % now run for new validation/test set Xval
    % Xval = Xval(1:100,:);yval = yval(1:100);
    allDist_test = slmetric_pw(X',mappedTestX','eucdist');

    % first row are just the points, then follow the nearest neighbors
    % for each test point, we look at the closest training points only
    [allPointsNNDistances_test, allPointsNN_test] = sort(allDist_test);
    S_test = allPointsNN_test(2:kNN+1,:);
    Sdist_test = allPointsNNDistances_test(2:kNN+1,:);
    sigma_test = sqrt(sum(allPointsNNDistances_test.^2)./kNN);
    pdist_test = lambda * sigma_test;
    Epdist_test = mean(pdistAll(categoryInd, S_test)); 
    plof_test = pdist_test./Epdist_test -1;
    loop_test = erf(plof_test./(nplofAll(categoryInd, :) * sqrt(2))); % use nplof from train
    loop_test(loop_test<0) = 0;
    
    probs(categoryInd, :) = loop_test;
end
finalProbs = min(probs);

figure;
fprintf('Visualizing test example dataset for outlier detection.\n\n');
hold on
plot(mappedTestX(:, 1), mappedTestX(:, 2), 'mx');
outliers_test = finalProbs > 0.9;

outliers_real = zeros(1, numImages);
for i = zeroCategories
    outliers_real = or(outliers_real, testY' == i);
end
plot(mappedTestX(outliers_test, 1), mappedTestX(outliers_test, 2), 'go', 'LineWidth', 2, 'MarkerSize', 10);
plot(mappedTestX(outliers_real, 1), mappedTestX(outliers_real, 2), 'bo', 'LineWidth', 2, 'MarkerSize', 10);
disp(['Found ' num2str(sum(outliers_test)) ' test outliers, ' num2str(sum(outliers_test==outliers_real)) '/' num2str(length(testY)) ' correct.'])
hold off