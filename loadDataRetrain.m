dataset = fullParams.dataset;
wordset = fullParams.wordset;
trainFrac = 1;

if not(exist('skipLoad','var')) || skipLoad == false
    disp('Loading data');
    load(['image_data/features/' dataset '/train.mat']);
    load(['image_data/features/' dataset '/test.mat']);
    load(['word_data/' wordset '/' dataset '/wordTable.mat']);
    skipLoad = true;
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
    numTrainTotalPerCat = numTrainNonZeroShot / length(nonZeroCategories);
    numTrainMapPerCat = floor(0.80 * numTrainTotalPerCat);
    numTrainOutlierPerCat = floor(0.15 * numTrainTotalPerCat);
    numValidatePerCat = numTrainTotalPerCat - numTrainMapPerCat - numTrainOutlierPerCat;
    t1 = zeros(1, numTrainMapPerCat * length(nonZeroCategories));
    t2 = zeros(1, numTrainOutlierPerCat * length(nonZeroCategories));
    v = zeros(1, numValidatePerCat * numCategories);
    for i = 1:length(nonZeroCategories)
        [ ~, temp ] = find(trainY == nonZeroCategories(i));
        t1((i-1)*numTrainMapPerCat+1:i*numTrainMapPerCat) = temp(1:numTrainMapPerCat);
        t2((i-1)*numTrainOutlierPerCat+1:i*numTrainOutlierPerCat) = temp(numTrainMapPerCat+1:numTrainMapPerCat+numTrainOutlierPerCat);
        v((i-1)*numValidatePerCat+1:i*numValidatePerCat) = temp(numTrainMapPerCat+numTrainOutlierPerCat+1:numTrainMapPerCat+numTrainOutlierPerCat+numValidatePerCat);
    end
    for i = 1:length(zeroCategories)
        [ ~, temp ] = find(trainY == zeroCategories(i));
        perm = randperm(length(temp));
        j = length(nonZeroCategories) + i;
        t2((j-1)*numTrainOutlierPerCat+1:j*numTrainOutlierPerCat) = temp(perm(1:numTrainOutlierPerCat));
        v((j-1)*numValidatePerCat+1:j*numValidatePerCat) = temp(perm(numTrainOutlierPerCat+1:numTrainOutlierPerCat+numValidatePerCat));
    end

    % permute
    order = randperm(numTrainMapPerCat * length(nonZeroCategories));
    t1 = t1(order);
    order = randperm(numTrainOutlierPerCat * numCategories);
    t2 = t2(order);
    order = randperm(numValidatePerCat * numCategories);
    v = v(order);
    XmapTrain = trainX(:, t1);
    YmapTrain = trainY(t1);
    XoutlierTrain = trainX(:, t2);
    YoutlierTrain = trainY(t2);
    Xvalidate = trainX(:, v);
    Yvalidate = trainY(v);
    save(sprintf('%s/perm.mat', outputPath), 't1', 't2', 'v');
end

fprintf('num map train: %d, num outlier train: %d, num valid: %d, num test: %d\n', length(YmapTrain), length(YoutlierTrain), length(Yvalidate), length(testY));
