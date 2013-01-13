function [guessedLabels] = feedforwardDiscriminant(thetaMapping, thetaSvm, trainParams, unseenWordTable, images, maxLogprobability, zeroCategoryTypes, nonzeroCategoryTypes, mu, sigma, priors)

[ W, b ] = stack2param(thetaMapping, trainParams.decodeInfo);

% Forward Propagation
mappedImages = bsxfun(@plus, 0.5 * W{1} * images, b{1});

logprobabilities = predictGaussianDiscriminant(mappedImages, mu, sigma, priors, zeroCategoryTypes);
unseenIndices = logprobabilities < maxLogprobability;
seenIndices = ~unseenIndices;

% If seen
[~, gind] = max(thetaSvm*images(:, seenIndices), [], 1);
guessedLabels(seenIndices) = nonzeroCategoryTypes(gind);

% This is the unseen label classifier
tDist = slmetric_pw(unseenWordTable, mappedImages(:, unseenIndices), 'eucdist');
[~, tGuessedCategories ] = min(tDist);
guessedLabels(unseenIndices) = zeroCategoryTypes(tGuessedCategories);

end
