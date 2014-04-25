% Load data (used in main.m)

dataset = fullParams.dataset;
wordset = fullParams.wordset;
trainFrac = 1;

if not(exist('skipLoad','var')) || skipLoad == false
    disp('Loading data');
    load(['image_data/features/' dataset '/train.mat']);
    load(['image_data/features/' dataset '/test.mat']);
    load(['word_data/' wordset '/' dataset '/wordTable.mat']);
end

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
elseif strcmp(dataset, 'animals')
    if not(exist('skipLoad','var')) || skipLoad == false
        disp('Loading data');
        load('image_data/images/animals/zero.mat');
        zeroCategories = arrayfun(@(x) find(ismember(label_names, zero_label_names{x})), 1:length(zero_label_names));
        
        % Mark zeroCategories
        numCategories = length(label_names);
        nonZeroCategories = setdiff(1:numCategories, zeroCategories);

        newTrain = [];
        newV1 = [];
        newTest = [];
        for i = 1:length(nonZeroCategories)
            currIdxs = find(trainY == nonZeroCategories(i));
            tids = randperm(length(currIdxs));
            train_id_cutoff = floor(0.9 * length(tids));
            newTrain = [ newTrain currIdxs(tids(1:train_id_cutoff)) ];
            newV1 = [ newV1 currIdxs(tids(train_id_cutoff+1:end)) ];
            newTest = [ newTest find(testY == nonZeroCategories(i)) ];
        end
        
        % add some unseen images to v for testing loOP model
        newV2 = [];
        for i = 1:length(zeroCategories)
            currIdxs = find(testY == zeroCategories(i));
            tids = randperm(length(currIdxs));
            test_id_cutoff = floor(0.9 * length(tids));
            newTest = [ newTest currIdxs(tids(1:test_id_cutoff)) ];
            newV2 = [ newV2 currIdxs(tids(test_id_cutoff+1:end)) ];
        end
        newTrain = newTrain(randperm(length(newTrain)));
        newTest = newTest(randperm(length(newTest)));
        
        X = trainX(:, newTrain);
        Y = trainY(newTrain);
        Xvalidate = [trainX(:, newV1) testX(:, newV2)];
        Yvalidate = [trainY(newV1) testY(newV2)];
        newV = randperm(length(Yvalidate));
        Xvalidate = Xvalidate(:, newV);
        Yvalidate = Yvalidate(newV);
        testX = testX(:, newTest);
        testY = testY(newTest);
        zeroList = label_names(zeroCategories);
        zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];
        numTrainPerCat = min(arrayfun(@(x) sum(Y == x), nonZeroCategories));
        outputPath = sprintf('gauss_%s_%s_%s', dataset, wordset, zeroStr);

        if not(exist(outputPath, 'dir'))
            mkdir(outputPath);
        end
        
        save([outputPath '/finalIds.mat'], 'newTrain', 'newTest', 'newV1', 'newV2');
        disp('Zero categories:');
        disp(zeroCategories);
    end    
end

fprintf('num train: %d, num valid: %d, num test: %d\n', length(Y), length(Yvalidate), length(testY));

% At the end, we have X, Y, Xvalidate, Yvalidate, wordTable, outputPath,
% numCategories, nonZeroCategories, zeroCategories, testX, testY,
% label_names, numTrainPerCat in the workspace
