function [ nplofAll, pdistAll ] = trainOutlierPriors(trainX, trainY, nonZeroCategories, numPerCategory, kNN, lambdas)

addpath toolbox/pwmetric;

nplofAll = zeros(length(nonZeroCategories), 1);
pdistAll = zeros(length(nonZeroCategories), numPerCategory);
for categoryInd = 1:length(nonZeroCategories)
    filtered = find(trainY == nonZeroCategories(categoryInd));
    X = trainX(:, filtered(randi(length(filtered), 1, numPerCategory)));
    
    %compute all nearest neighbors
    allDist = slmetric_pw(X, X,'eucdist');

    %first row are just the points, then follow the nearest neighbors
    [allPointsNNDistances, allPointsNN] = sort(allDist);

    S = allPointsNN(2:kNN+1,:);
    sigma = sqrt(sum(allPointsNNDistances.^2)./kNN);
    pdist = lambdas(categoryInd) * sigma;
    Epdist = mean(pdist(S));
    plof = pdist./Epdist -1;
    nplof = lambdas(categoryInd) * sqrt(mean(plof.^2));
    
    nplofAll(categoryInd, :) = nplof;
    pdistAll(categoryInd, :) = pdist;
end

end