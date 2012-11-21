function [] = test( outputPath, batchFilePrefix )

addpath toolbox/;

load(sprintf('%s/params_final.mat', outputPath), 'theta', 'trainParams');

% Additional options
disp('Loading test images');
batchFilePath   = 'image_data/cifar-10-features';
[imgs, categories, categoryNames] = loadTrainBatch(batchFilePrefix, batchFilePath);

%% Load word representations
disp('Loading word representations');
ee = load(['word_data/' trainParams.wordDataset '/embeddings.mat']);
vv = load(['word_data/' trainParams.wordDataset '/vocab.mat']);
trainParams.embeddingSize = size(ee.embeddings, 1);
wordTable = zeros(trainParams.embeddingSize, length(categoryNames));
for categoryIndex = 1:length(categoryNames)
    icategoryWord = find(ismember(vv.vocab, categoryNames(categoryIndex)) == true);
    wordTable(:, categoryIndex) = ee.embeddings(:, icategoryWord);
end
clear ee vv;

disp('Test results');
doEvaluate(imgs, categories, categoryNames, wordTable, theta, trainParams);

end
