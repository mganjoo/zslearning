function [ results ] = softmaxDoEvaluate( images, categories, categoryNames, theta, trainParams, doPrint )

W = stack2param(theta, trainParams.decodeInfo);
numCategories = length(categoryNames);
numImages = size(images, 2);

pred = exp(W{1}*images); % k by n matrix with all calcs needed
pred = bsxfun(@rdivide,pred,sum(pred));
[~, guessedCategories] = max(pred);

% Calculate scores
confusion = zeros(numCategories, numCategories);
for actual = 1:numCategories
    guessesForCategory = guessedCategories(categories == actual);
    for guessed = 1:numCategories
        confusion(actual, guessed) = sum(guessesForCategory == guessed);
    end
end

truePos = diag(confusion); % true positives, column vector
results.accuracy = sum(truePos) / numImages;
t = truePos ./ sum(confusion, 2);
results.avgPrecision = mean(t(isfinite(t), :));
t = truePos' ./ sum(confusion, 1);
results.avgRecall = mean(t(:, isfinite(t)));

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, categoryNames);
end

end

