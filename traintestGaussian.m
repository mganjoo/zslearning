addpath anomalyFunctions/;
addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

dataset = 'cifar10';
wordset = 'acl';
trainFrac = 0.8;

if strcmp(dataset, 'cifar10')
    TOTAL_NUM_TRAIN = 50000;
    TOTAL_NUM_PER_CATEGORY = 5000;
    numCategories = 10;
    % 'cat', 'truck'
    zeroCategories = [ 4, 10 ];
else
    TOTAL_NUM_TRAIN = 48000;
    TOTAL_NUM_PER_CATEGORY = 500;
    numCategories = 96;
    % 'boy', 'lion', 'orange', 'train', 'couch', 'house' 
    zeroCategories = [ 12, 42, 52, 87, 26, 36 ];
end
outputPath = sprintf('gauss_%s_%s', dataset, wordset);

if not(exist('skipLoad','var')) || skipLoad == false
    disp('Loading data');
    load(['image_data/features/' dataset '/train.mat']);
    load(['image_data/features/' dataset '/test.mat']);
    load(['word_data/' wordset '/' dataset '/wordTable.mat']);
end
    
if not(exist(outputPath, 'dir'))
    mkdir(outputPath);
end

disp('Zero categories:');
disp(zeroCategories);
nonZeroCategories = setdiff(1:numCategories, zeroCategories);

numTrain = (numCategories - length(zeroCategories)) / numCategories * TOTAL_NUM_TRAIN;
numTrain1 = round(trainFrac * numTrain);
numTrain2 = numTrain - numTrain1;
numTrainPerCat1 = numTrain1 / length(nonZeroCategories);
numTrainPerCat2 = numTrain2 / length(nonZeroCategories);

% divide into two training sets (for mapping function and threshold) 
X1 = zeros(size(trainX, 1), numTrain1);
X2 = zeros(size(trainX, 1), numTrain2);
Y1 = zeros(1, numTrain1);
Y2 = zeros(1, numTrain2);
for i = 1:length(nonZeroCategories)
    [ ~, t ] = find(trainY == nonZeroCategories(i));
    X1(:, (i-1)*numTrainPerCat1+1:i*numTrainPerCat1) = trainX(:, t(1:numTrainPerCat1));
    X2(:, (i-1)*numTrainPerCat2+1:i*numTrainPerCat2) = trainX(:, t(numTrainPerCat1+1:end));
    Y1((i-1)*numTrainPerCat1+1:i*numTrainPerCat1) = trainY(t(1:numTrainPerCat1));
    Y2((i-1)*numTrainPerCat2+1:i*numTrainPerCat2) = trainY(t(numTrainPerCat1+1:end));
end

% permute
order1 = randperm(numTrain1);
order2 = randperm(numTrain2);
X1 = X1(:, order1);
X2 = X2(:, order2);
Y1 = Y1(order1);
Y2 = Y2(order2);

% disp('Training mapping function');
% % Train mapping function
% fastTrain;
% save(sprintf('%s/theta.mat', outputPath), 'theta', 'trainParams');
% 
% disp('Training SVM features');
% % Train SVM features
% L = 0.01;
% mappedCategories = zeros(1, numCategories);
% mappedCategories(nonZeroCategories) = 1:numCategories-length(zeroCategories);
% thetaSvm = train_svm(X1', mappedCategories(Y1)', 1/L)';
% save(sprintf('%s/thetaSvm.mat', outputPath), 'thetaSvm');

disp('Training Gaussian classifier');
% Train Gaussian classifier
[ W, b ] = stack2param(theta, trainParams.decodeInfo);
mapped = bsxfun(@plus, 0.5 * W{1} * X2, b{1});
[mu, sigma, priors] = trainGaussianDiscriminant(mapped, Y2, numCategories, wordTable);
sortedLogprobabilities = sort(predictGaussianDiscriminant(mapped, mu, sigma, priors, zeroCategories));

% Test
seenAccuracies = numTrain2 / 100;
unseenAccuracies = numTrain2 / 100;

for i = 1:numTrain2/100
    cutoff = sortedLogprobabilities((i-1)*100+1);

    % Test Gaussian classifier
    results = mapGaussianThresholdDoEvaluate( testX, testY, zeroCategories, label_names, wordTable, ...
        theta, thetaSvm, trainParams, cutoff, mu, sigma, priors, false);

    seenAccuracies(i) = results.seenAccuracy;
    unseenAccuracies(i) = results.unseenAccuracy;

end
zeroList = label_names(zeroCategories);
zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];
save(sprintf('%s/out_%s.mat', outputPath, zeroStr), 'results', 'seenAccuracies', 'unseenAccuracies');
