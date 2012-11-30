function [ guessedCategoriesDebug, results ] = test( filename, batchFilePrefix, dataset )

addpath toolbox/;

t = load(filename, 'theta', 'trainParams');

% Additional options
disp('Loading test images');
[imgs, categories, originalCategoryNames] = loadBatch(batchFilePrefix, dataset);

%% Load word representations
testCategoryNames = loadCategoryNames({ 'lion', 'orange', 'camel' }, dataset);
w = load(['word_data/' t.trainParams.wordDataset '/wordTable.mat']);
trainParams.embeddingSize = size(w.wordTable, 1);
wordTable = zeros(trainParams.embeddingSize, length(testCategoryNames));
for categoryIndex = 1:length(testCategoryNames)
    icategoryWord = ismember(w.label_names, testCategoryNames(categoryIndex)) == true;
    wordTable(:, categoryIndex) = w.wordTable(:, icategoryWord);
end
clear w;

disp('Test results');
[ guessedCategoriesDebug, results ] = doEvaluate(imgs, categories, originalCategoryNames, testCategoryNames, wordTable, t.theta, t.trainParams);

end
