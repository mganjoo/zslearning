distMultipliers = [ 3 5 8 10 15 20 ];
numAdditional = [ 1 2 3 5 7 9 ];

for i = 1:length(distMultipliers)
    for j = 1:length(numAdditional)
        trainParams.distanceMultiplier = distMultipliers(i);
        trainParams.numAdditionalCategories = numAdditional(j);
        fprintf('Dist: %f NumAdditional: %f\n', trainParams.distanceMultiplier, trainParams.numAdditionalCategories);
        mapConvexHullDoEvaluate( X, Y, [4, 10], label_names, wordTable, theta, trainParams, true );
    end
end
