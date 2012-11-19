function [ guessedCategoriesDebug, accuracy ] = doTest( images, categories, categoryLabels, wordTable, theta, trainParams )

[Wt, bt] = stack2param(theta, trainParams.decodeInfo);
numImages = size(images, 2);
numCategories = size(wordTable, 2);
guessedCategoriesDebug = zeros(size(categoryLabels, 1), numImages);
confusion = zeros(length(categoryLabels), length(categoryLabels));
numCorrect = 0;

% For each category, run and pick max score
for j = 1:numCategories
    p = [repmat(wordTable(:, j), 1, numImages); images];
    scores = Wt{2} *  trainParams.f(bsxfun(@plus, Wt{1} * p, bt{1}));
    guessedCategoriesDebug(j, :) = scores;
end

[ ~, guessedCategories ] = max(guessedCategoriesDebug);

for actual = 1:numCategories
    guessesForCateg = guessedCategories(categories == actual);
    for guessed = 1:numCategories
        confusion(actual, guessed) = sum(guessesForCateg == guessed);
    end
    numCorrect = numCorrect + confusion(actual, actual);
end

accuracy = numCorrect / numImages;
disp(['Accuracy: ' num2str(accuracy)]);
displayConfusionMatrix(confusion, categoryLabels);

end

