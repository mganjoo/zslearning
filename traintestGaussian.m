addpath anomalyFunctions/;
addpath toolbox/;
addpath toolbox/minFunc/;
addpath toolbox/pwmetric/;
addpath costFunctions/;

numCategories = 10;
outputPath = 'allExecs';

load('image_data/features/cifar10/train.mat');
load('image_data/features/cifar10/test.mat');
load('word_data/acl/cifar10/wordTable.mat');
load('image_data/images/meta.mat');

if not(exist(outputPath, 'dir'))
    mkdir(outputPath);
end

for z1 = 1:numCategories-1
    for z2 = z1+1:numCategories
        zeroCategories = [z1 z2];
        nonZeroCategories = setdiff(1:numCategories, zeroCategories);
        
        % divide into two training sets (for mapping function and threshold) 
        X1 = zeros(size(trainX, 1), 32000);
        X2 = zeros(size(trainX, 1), 8000);
        Y1 = zeros(1, 32000);
        Y2 = zeros(1, 8000);
        for i = 1:length(nonZeroCategories)
            [ ~, t ] = find(trainY == nonZeroCategories(i));
            X1(:, (i-1)*4000+1:i*4000) = trainX(:, t(1:4000));
            X2(:, (i-1)*1000+1:i*1000) = trainX(:, t(4001:end));
            Y1((i-1)*4000+1:i*4000) = tempTrainY(t(1:4000));
            Y2((i-1)*1000+1:i*1000) = tempTrainY(t(4001:end));
        end
        
        % permute
        order1 = randperm(32000);
        order2 = randperm(8000);
        X1 = X1(order1);
        X2 = X2(order2);
        Y1 = Y1(order1);
        Y2 = Y2(order2);
        
        % Train mapping function
        fastTrain;
        
        % Train SVM features
        L = 0.01;
        thetaSvm = train_svm(X1', Y1', 1/L);
        
        % Train Gaussian classifier
        [ W, b ] = stack2param(theta, trainParams.decodeInfo);
        mapped = bsxfun(@plus, 0.5 * W{1} * X2, b{1});
        [mu, sigma, priors] = trainGaussianDiscriminant(mapped, Y2, numCategories, wordTable);
        sortedLogprobabilities = sort(predictGaussianDiscriminant(mapped, mu, sigma, priors, zeroCategories));

        % Test
        seenAccuracies = zeros(1, 80);
        unseenAccuracies = zeros(1, 80);

        for i = 1:80
            cutoff = sortedLogprobabilities((i-1)*100+1);

            % Test Gaussian classifier
            results = mapGaussianThresholdDoEvaluate( testX, testY, zeroCategories, label_names, wordTable, ...
                theta, thetaSvm, trainParams, cutoff, mu, sigma, priors, false);

            seenAccuracies(i) = results.seenAccuracy;
            unseenAccuracies(i) = results.unseenAccuracy;
            
            save(sprintf('%s/out_%s_%s.mat', outputPath, label_names{z1}, label_names{z2}), 'results', 'seenAccuracies', 'unseenAccuracies');
        end        
    end
end
