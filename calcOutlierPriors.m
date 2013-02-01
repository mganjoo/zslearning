function [ probs ] = calcOutlierPriors( testX, trainX, lambda, kNN, nplof, pdist )

allDist_test = slmetric_pw(trainX, testX, 'eucdist');
[allPointsNNDistances_test, allPointsNN_test] = sort(allDist_test);
S_test = allPointsNN_test(2:kNN+1,:);
sigma_test = sqrt(sum(allPointsNNDistances_test.^2)./kNN);
pdist_test = lambda * sigma_test;
Epdist_test = mean(pdist(S_test)); 
plof_test = pdist_test./Epdist_test -1;
probs = erf(plof_test./(nplof * sqrt(2))); % use nplof from train
probs(probs<0) = 0;

end

