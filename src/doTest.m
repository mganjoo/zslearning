function [ guessedCategoriesDebug, accuracy ] = doTest( images, categories, categoryLabels, wordTable, theta, trainParams )

[Wt, bt] = stack2param(theta, trainParams.decodeInfo);
numImages = size(images, 2);
numCategories = size(wordTable, 2);
guessedCategories = zeros(size(categories));
guessedCategoriesDebug = zeros(size(categories, 1), size(categoryLabels, 1) + 2);
confusion = zeros(length(categoryLabels), length(categoryLabels));
numCorrect = 0;
for i = 1:numImages
    maxscore = -Inf;
    % For each category, run and pick max score
    for j = 1:numCategories
        p = [wordTable(:, j); images(:, i)];
        score = Wt{2} *  trainParams.f(bsxfun(@plus, Wt{1} * p, bt{1}));
        guessedCategoriesDebug(i, j) = score;
        if score > maxscore
            maxscore = score;
            guessedCategories(i) = j;
        end
    end
    actual = categories(i);
    guessed = guessedCategories(i);
    guessedCategoriesDebug(i, end-1:end) = [ actual guessed ];
    confusion(actual, guessed) = confusion(actual, guessed) + 1;
    if actual == guessed
        numCorrect = numCorrect + 1;
    end
end

accuracy = numCorrect / numImages;
disp(['Accuracy: ' num2str(accuracy)]);
displayConfusionMatrix(confusion, categoryLabels);

end

