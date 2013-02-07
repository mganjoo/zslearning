function [ finalProbs ] = calcOutlierPriors( testX, trainX, trainY, numPerCategory, nonZeroCategories, lambdas, kNN, nplofAll, pdistAll )

numImages = size(testX, 2);
numNonZeroCategories = length(nonZeroCategories);
probs = zeros(numNonZeroCategories, numImages);
for categoryInd = 1:numNonZeroCategories
    filtered = find(trainY == nonZeroCategories(categoryInd));
    X = trainX(:, filtered(randi(length(filtered), 1, numPerCategory)));
    allDist_test = slmetric_pw(X,testX,'eucdist');
    [allPointsNNDistances_test, allPointsNN_test] = sort(allDist_test);
    S_test = allPointsNN_test(2:kNN+1,:);
    sigma_test = sqrt(sum(allPointsNNDistances_test.^2)./kNN);
    pdist_test = lambdas(categoryInd) * sigma_test;
    Epdist_test = mean(pdistAll(categoryInd, S_test)); 
    plof_test = pdist_test./Epdist_test -1;
    loop = erf(plof_test./(nplofAll(categoryInd, :) * sqrt(2))); % use nplof from train
    loop(loop<0) = 0;
    probs(categoryInd, :) = loop;
end
finalProbs = min(probs);


end

