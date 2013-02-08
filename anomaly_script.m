addpath toolbox/;
addpath anomalyFunctions/;

fields = {{'numCategories',       10};
          {'zeroCategories',      [4, 10]};
};

% Load existing model parameters, if they exist
for i = 1:length(fields)
    if exist('anomalyParams','var') && isfield(anomalyParams,fields{i}{1})
        disp(['Using the previously defined parameter ' fields{i}{1}])
    else
        anomalyParams.(fields{i}{1}) = fields{i}{2};
    end
end

numCategories = anomalyParams.numCategories;
zeroCategories = anomalyParams.zeroCategories;
knn = 20;
nonZeroCategories = setdiff(1:numCategories, zeroCategories);

% Load images
load('image_data/batches/cifar10/validation_both.mat');

% Load trained mapped images
load('mappedTrainData.mat');
numImages = length(trainY);
numPerCategory = 3950;

% Load other parameters
t = load('map-goodParams/params_final.mat');
thetaMapping = t.theta;
trainParams = t.trainParams;
[ W, b ] = stack2param(thetaMapping, trainParams.decodeInfo);
mappedImages = bsxfun(@plus, 0.5 * W{1} * X, b{1});

t = load('map-softmaxGoodParams/params_final.mat');
thetaSeenSoftmax = t.theta;
seenTrainParams = t.trainParams;

t = load('map-unseenSoftmaxGoodParams/params_final.mat');
thetaUnseenSoftmax = t.theta;
unseenTrainParams = t.trainParams;

% Load unseen word table
t = load('word_data/acl/cifar10/wordTable.mat');
load('image_data/images/cifar10/meta.mat');
wordTable = t.wordTable;
unseenWordTable = t.wordTable(:, zeroCategories);
clear t;

% Train and test anomaly detector
resolution = 20;
thresholds = 0:(1/resolution):1;
% lambdas = 1:12;
% seenAccuracies = zeros(length(lambdas), length(thresholds));
% unseenAccuracies = zeros(length(lambdas), length(thresholds));
% accuracies = zeros(length(lambdas), length(thresholds));
% nonZeroCategoryIdPerm = randperm(length(nonZeroCategories));
% bestLambdas = repmat(lambdas(length(lambdas)/2), 1, length(nonZeroCategories));
% for k = 1:length(nonZeroCategories)
%     changedCategory = nonZeroCategoryIdPerm(k);
%     for i = 1:length(lambdas)
%         tempLambdas = bestLambdas;
%         tempLambdas(changedCategory) = lambdas(i);
%         disp(tempLambdas);
%         [ nplofAll, pdistAll ] = trainOutlierPriors(trainX, trainY, nonZeroCategories, numPerCategory, knn, tempLambdas);
%         probs = calcOutlierPriors( mappedImages, trainX, trainY, numPerCategory, nonZeroCategories, tempLambdas, knn, nplofAll, pdistAll );
%         for t = 1:length(thresholds)
%             fprintf('Threshold %f: ', thresholds(t));
%             [~, results] = anomalyDoEvaluate(thetaSeenSoftmax, ...
%                 seenTrainParams, thetaUnseenSoftmax, unseenTrainParams, probs, X, mappedImages, Y, ...
%                 thresholds(t), zeroCategories, nonZeroCategories, false);
%             seenAccuracies(i, t) = results.seenAccuracy;
%             unseenAccuracies(i, t) = results.unseenAccuracy;
%             accuracies(i, t) = results.accuracy;
%             fprintf('seen accuracy: %f, unseen accuracy: %f\n', results.seenAccuracy, results.unseenAccuracy);
%         end
%     end
%     [~, t] = max(sum(accuracies,2));
%     bestLambdas(k) = t;
% end
% disp('Best:');
% disp(bestLambdas);

% % Do it again, with best lambdas
% seenAccuracies = zeros(1, length(thresholds));
% unseenAccuracies = zeros(1, length(thresholds));
% accuracies = zeros(1, length(thresholds));
% [ nplofAll, pdistAll ] = trainOutlierPriors(trainX, trainY, nonZeroCategories, numPerCategory, knn, bestLambdas);
% probs = calcOutlierPriors( mappedImages, trainX, trainY, numPerCategory, nonZeroCategories, bestLambdas, knn, nplofAll, pdistAll );
% for t = 1:length(thresholds)
%     fprintf('Threshold %f: ', thresholds(t));
%     [~, results] = anomalyDoEvaluate(thetaSeenSoftmax, ...
%         seenTrainParams, thetaUnseenSoftmax, unseenTrainParams, probs, X, mappedImages, Y, ...
%         thresholds(t), zeroCategories, nonZeroCategories, false);
%     seenAccuracies(t) = results.seenAccuracy;
%     unseenAccuracies(t) = results.unseenAccuracy;
%     accuracies(t) = results.accuracy;
%     fprintf('accuracy: %f, seen accuracy: %f, unseen accuracy: %f\n', results.accuracy, results.seenAccuracy, results.unseenAccuracy);
% end

% Train and test Gaussian anomaly detector
[mu, sigma, priors] = trainGaussianDiscriminant(trainX, trainY, numCategories, wordTable);
sortedLogprobabilities = sort(predictGaussianDiscriminantMin(trainX, mu, sigma, priors, zeroCategories));

% Test
numTrain = size(trainX, 2);
resolution = resolution+1; %%%%%%%%%
oseenAccuracies = zeros(1, resolution);
ounseenAccuracies = zeros(1, resolution);
oAccuracies = zeros(1, resolution);
divFactor = floor(numTrain / resolution);
for i = 1:resolution
    cutoff = sortedLogprobabilities((i-1)*divFactor+1);
    fprintf('Cutoff %f: ', cutoff);
    % Test Gaussian classifier
    results = mapGaussianThresholdDoEvaluate( X, Y, zeroCategories, label_names, wordTable, ...
        thetaMapping, trainParams, thetaSeenSoftmax, seenTrainParams, thetaUnseenSoftmax, unseenTrainParams, cutoff, mu, sigma, priors, false);
    oseenAccuracies(i) = results.seenAccuracy;
    ounseenAccuracies(i) = results.unseenAccuracy;
    oAccuracies(i) = results.accuracy;
    fprintf('accuracy: %f, seen accuracy: %f, unseen accuracy: %f\n', results.accuracy, results.seenAccuracy, results.unseenAccuracy);
end
oseenAccuracies = fliplr(oseenAccuracies);
ounseenAccuracies = fliplr(ounseenAccuracies);
oAccuracies = fliplr(oAccuracies);

hold on;
ColorSet = varycolor(3);
plot(thresholds, seenAccuracies, 'Color', ColorSet(1,:));
plot(thresholds, unseenAccuracies, 'Color', ColorSet(2,:));
plot(thresholds, accuracies, 'Color', ColorSet(3,:));
plot(thresholds, oseenAccuracies, '--', 'Color', ColorSet(1,:));
plot(thresholds, ounseenAccuracies, '--', 'Color', ColorSet(2,:));
plot(thresholds, oAccuracies, '--', 'Color', ColorSet(3,:));
plot(thresholds, oseenAccuraciesNew, ':', 'Color', ColorSet(1,:));
plot(thresholds, ounseenAccuraciesNew, ':', 'Color', ColorSet(2,:));
plot(thresholds, oAccuraciesNew, ':', 'Color', ColorSet(3,:));
legend({ 'loop model seen', 'loop model unseen', 'loop model total',...
    'gaussian mix model seen', 'gaussian mix model unseen', 'gaussian mix model total',...
    'gaussian pdf model seen', 'gaussian pdf model unseen', 'gaussian pdf model total'});

% ColorSet = varycolor(11);
% hold on;
% leg = cell(1, length(lambdas));
% for i = 1:length(lambdas)
%     plot(thresholds, seenAccuracies(i, :), 'Color', ColorSet(i,:));
%     leg{i} = num2str(lambdas(i));
% end
% legend(leg);
% for i = 1:length(lambdas)
%     plot(thresholds, unseenAccuracies(i, :), 'Color', ColorSet(i,:));
% end
% plot(thresholds, [oseenAccuracies; ounseenAccuracies], '--', 'Color', ColorSet(length(lambdas)+1,:));
