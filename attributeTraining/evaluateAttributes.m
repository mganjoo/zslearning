% Evaluate attributes model (Lampert et al.)

function [ guessedCategories, results ] = evaluateAttributes( X, Y, thetas, trainParams, ...
    assignments, categoriesToConsider, nonZeroCategories, categoryNames, doPrint)
    numCategories = length(categoriesToConsider);
    numAttributes = length(thetas);
    
    X = X(:, ismember(Y, categoriesToConsider));
    Y = Y(ismember(Y, categoriesToConsider));
    
    numImages = length(Y);
    P = zeros(numAttributes, numImages); % (85, N)
    for i = 1:numAttributes
        W = stack2param(thetas{i}, trainParams{i}.decodeInfo);
        pred = exp(W{1}*X); % k by n matrix with all calcs needed
        P(i, :) = bsxfun(@rdivide, pred, sum(pred));
    end
    
    % Priors
    assignments = assignments';
    prior = mean(assignments(nonZeroCategories, :)); % (1, 85)
    prior(prior==0.) = 0.5;
    prior(prior==1.) = 0.5; % disallow degenerate priors

    M = assignments(categoriesToConsider, :); % (n_t, 85)
    probs = zeros(numCategories, numImages);
    denom = prod(bsxfun(@times, M, prior) + bsxfun(@times, 1-M, 1-prior), 2);
    for i = 1:numImages
        probs(:, i) = prod(bsxfun(@times, M, P(:, i)') + bsxfun(@times, 1-M, 1-P(:, i)'), 2) / denom;
    end
    [~, guessedCategories] = max(probs);

    % Calculate scores
    confusion = zeros(numCategories, numCategories);
    for actual = 1:numCategories
        guessesForCategory = guessedCategories(Y == actual);
        for guessed = 1:numCategories
            confusion(actual, guessed) = sum(guessesForCategory == guessed);
        end
    end

    truePos = diag(confusion); % true positives, column vector
    results.confusion = confusion;
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

