function [ results ] = mapRandomWordDoEvaluate( images, categories, zeroCategoryTypes, categoryNames, wordTable, thetaMapping, thetaAnomaly, thetaSvm, trainParams, doPrint )

addpath toolbox/pwmetric;

[ W, b ] = stack2param(thetaMapping, trainParams.mappingDecodeInfo);

% Build unseen word table
ee = load(['word_data/' trainParams.wordDataset '/embeddings.mat']);
vv = load(['word_data/' trainParams.wordDataset '/vocab.mat']);
existingIndices = arrayfun(@(x) ismember(x, vv.vocab), categoryNames(zeroCategoryTypes));
diffset = setdiff(1:length(vocab), existingIndices);
randCategoryIndices = randi(length(diffset), 1, trainParams.numExtra);
newWordTable = ee.embeddings(:, [ existingIndices randCategoryIndices ]);
clear ee vv;

keep = arrayfun(@(x) ~ismember(x, zeroCategoryTypes), 1:length(categoryNames));
unseenWordTable = wordTable(:, ~keep);
nonzeroCategories = setdiff(1:length(categoryNames), zeroCategoryTypes);
guessedCategories = zeros(size(categories));

numImages = size(images, 2);
numCategories = size(wordTable, 2);

disp('Loading random words for evaluation');

% Keep only zeroshot images
zeroIndices = ismember(categories, zeroCategoryTypes);
images = images(zeroIndices);
categories = categories(zeroIndices);

% Feedforward
mappedImages = bsxfun(@plus, 0.5 * W{1} * images, b{1});

tDist = slmetric_pw(newWordTable, mappedImages, 'eucdist');
[~, tGuessedCategories ] = min(tDist);
% if the category is not correct, we don't care what it is
tGuessedCategories(tGuessedCategories > length(zeroCategoryTypes)) = -1;


guessedCategories(unseenIndices) = zeroCategoryTypes(tGuessedCategories);

% Calculate scores
confusion = zeros(numCategories, numCategories);
for actual = 1:numCategories
    guessesForCategory = guessedCategories(categories == actual);
    for guessed = 1:numCategories
        confusion(actual, guessed) = sum(guessesForCategory == guessed);
    end
end

truePos = diag(confusion); % true positives, column vector
results.accuracy = sum(truePos) / numImages;
numUnseen = sum(arrayfun(@(x) nnz(categories == x), zeroCategoryTypes));
results.unseenAccuracy = sum(truePos(zeroCategoryTypes)) / numUnseen;
results.seenAccuracy = (sum(truePos) - sum(truePos(zeroCategoryTypes))) / (numImages - numUnseen);
t = truePos ./ sum(confusion, 2);
results.avgPrecision = mean(t(isfinite(t), :));
t = truePos' ./ sum(confusion, 1);
results.avgRecall = mean(t(:, isfinite(t)));

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Seen Accuracy: ' num2str(results.seenAccuracy)]);
    disp(['Unseen Accuracy: ' num2str(results.unseenAccuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, categoryNames);
end

end

