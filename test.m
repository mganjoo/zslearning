function [ guessedCategoriesDebug, results ] = test( filename, batchFilePrefix, dataset )

addpath toolbox/;

t = load(filename, 'theta', 'trainParams');

% Additional options
disp('Loading test images');
batchFilePath   = 'image_data/cifar-features';
[imgs, categories, originalCategoryNames] = loadBatch(batchFilePrefix, batchFilePath);

%% Load word representations
disp('Loading word representations');
ee = load(['word_data/' t.trainParams.wordDataset '/embeddings.mat']);
vv = load(['word_data/' t.trainParams.wordDataset '/vocab.mat']);

assert((strcmp(dataset, 'cifar10') == true) || (strcmp(dataset, 'cifar100') == true));
if strcmp(dataset, 'cifar10') == true
    testCategoryNames = loadCategoryNames({ 'truck' }, dataset);
else
    testCategoryNames = loadCategoryNames({ 'lion', 'orange', 'camel' }, dataset);
end
t.trainParams.embeddingSize = size(ee.embeddings, 1);
wordTable = zeros(t.trainParams.embeddingSize, length(testCategoryNames));
for categoryIndex = 1:length(testCategoryNames)
    icategoryWord = ismember(vv.vocab, testCategoryNames(categoryIndex)) == true;
    wordTable(:, categoryIndex) = ee.embeddings(:, icategoryWord);
end
clear ee vv f;

disp('Test results');
[ guessedCategoriesDebug, results ] = doEvaluate(imgs, categories, originalCategoryNames, testCategoryNames, wordTable, t.theta, t.trainParams);

end
