% Load data (used in main.m)
% Prerequisite: feature files for any dataset you'd
% like to use must be present in image_data/features
% (both train.mat and test.mat). There must also be a
% wordTable.mat present in word_data/<wordset_name>/<dataset_name>.

dataset = fullParams.dataset;
wordset = fullParams.wordset;
trainFrac = 1;

if not(exist('skipLoad','var')) || skipLoad == false
    disp('Loading data');
    load(['image_data/features/' dataset '/train.mat']);
    load(['image_data/features/' dataset '/test.mat']);
    load(['word_data/' wordset '/' dataset '/wordTable.mat']);
end

% Split data
if strcmp(dataset, 'cifar10') || strcmp(dataset, 'cifar96') || strcmp(dataset, 'cifar106')
    if strcmp(dataset, 'cifar10')
        TOTAL_NUM_TRAIN = 50000;
        TOTAL_NUM_PER_CATEGORY = 5000;
        numCategories = 10;
        if isfield(fullParams,'zeroCategories')
            zeroCategories = fullParams.zeroCategories;
        else
            % 'cat', 'truck'
            zeroCategories = [ 4, 10 ];
        end
    elseif strcmp(dataset, 'cifar96')
        TOTAL_NUM_TRAIN = 48000;
        TOTAL_NUM_PER_CATEGORY = 500;
        numCategories = 96;
        if isfield(fullParams,'zeroCategories')
            zeroCategories = fullParams.zeroCategories;
        else
            % 'boy', 'lion', 'orange', 'train', 'couch', 'house' 
            zeroCategories = [ 12, 42, 52, 87, 26, 36 ];
        end
    else
        TOTAL_NUM_TRAIN = 53000;
        TOTAL_NUM_PER_CATEGORY = 500;
        numCategories = 106;
        if isfield(fullParams,'zeroCategories')
            zeroCategories = fullParams.zeroCategories;
        else
            % 'forest', 'lobster', 'boy', 'truck', 'orange', 'cat'
            zeroCategories = [ 33, 44, 12, 106, 52, 100 ];
        end
    end

    zeroList = label_names(zeroCategories);
    zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];
    outputPath = sprintf('gauss_%s_%s_%s', dataset, wordset, zeroStr);

    if not(exist(outputPath, 'dir'))
        mkdir(outputPath);
    end

    disp('Zero categories:');
    disp(zeroCategories);
    nonZeroCategories = setdiff(1:numCategories, zeroCategories);

    numTrainNonZeroShot = (numCategories - length(zeroCategories)) / numCategories * TOTAL_NUM_TRAIN;
    numTrainPerCat = 0.95 * numTrainNonZeroShot / length(nonZeroCategories);
    numValidatePerCat = numTrainPerCat * 0.05 / 0.95;
    t = zeros(1, numTrainPerCat * length(nonZeroCategories));
    newV = zeros(1, numValidatePerCat * numCategories);
    for i = 1:length(nonZeroCategories)
        [ ~, temp ] = find(trainY == nonZeroCategories(i));
        t((i-1)*numTrainPerCat+1:i*numTrainPerCat) = temp(1:numTrainPerCat);
        newV((i-1)*numValidatePerCat+1:i*numValidatePerCat) = temp(numTrainPerCat+1:end);
    end
    for i = 1:length(zeroCategories)
        [ ~, temp ] = find(trainY == zeroCategories(i));
        j = length(nonZeroCategories) + i;
        newV((j-1)*numValidatePerCat+1:j*numValidatePerCat) = temp(1:numValidatePerCat);
    end

    % permute
    order = randperm(numTrainPerCat * length(nonZeroCategories));
    t = t(order);
    order = randperm(numValidatePerCat * numCategories);
    newV = newV(order);
    X = trainX(:, t);
    Y = trainY(t);
    Xvalidate = trainX(:, newV);
    Yvalidate = trainY(newV);
    save(sprintf('%s/perm.mat', outputPath), 't', 'newV');
end

fprintf('num train: %d, num valid: %d, num test: %d\n', length(Y), length(Yvalidate), length(testY));

% At the end, we have X, Y, Xvalidate, Yvalidate, wordTable, outputPath,
% numCategories, nonZeroCategories, zeroCategories, testX, testY,
% label_names, numTrainPerCat in the workspace
