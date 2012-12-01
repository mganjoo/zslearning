function [ guessedCategoriesDebug, results ] = mapTest( filename, batchFilePrefix, dataset )

addpath toolbox/;

t = load(filename, 'theta', 'trainParams');

% Additional options
disp('Loading test images');
[imgs, categories, originalCategoryNames] = loadBatch(batchFilePrefix, dataset);

%% Load word representations
if strcmp(dataset, 'cifar10') == true
    testCategoryNames = loadCategoryNames({ 'truck' }, dataset);
elseif strcmp(dataset, 'cifar96') == true
    testCategoryNames = loadCategoryNames({ 'lion', 'orange', 'camel' }, dataset);
else
    error('Not a valid dataset');
end
w = load(['word_data/' t.trainParams.wordDataset '/' dataset '/wordTable.mat']);
trainParams.embeddingSize = size(w.wordTable, 1);
wordTable = zeros(trainParams.embeddingSize, length(testCategoryNames));
for categoryIndex = 1:length(testCategoryNames)
    icategoryWord = ismember(w.label_names, testCategoryNames(categoryIndex)) == true;
    wordTable(:, categoryIndex) = w.wordTable(:, icategoryWord);
end
clear w;

disp('Test results');
[ guessedCategoriesDebug, results ] = mapDoEvaluate(imgs, categories, originalCategoryNames, testCategoryNames, wordTable, t.theta, t.trainParams);

end
