function [guessedLabels, confusionMatrix] = computeLabelsWordVectors(feedForward, theta, parameters, imageFeatures, labels, wordVectors)

%% STEP 1: Feedforward Autoencoder
guessedLabels = feedForward(theta, parameters, wordVectors, imageFeatures);

%% STEP 2: Confusion matrix

numLabels = size(wordVectors, 2);
confusionMatrix = zeros(numLabels, numLabels);
for i=1:size(labels, 1)
    confusionMatrix(labels(i), guessedLabels(i)) = confusionMatrix(labels(i), guessedLabels(i)) + 1;
end

end
