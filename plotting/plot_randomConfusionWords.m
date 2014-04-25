% Generates confusion words not already present in the dataset.
% Relevant section in paper: "6.7 Zero-Shot Classes with Distractor Words

% Instructions:
% 1. Load test.mat into workspace.
% 2. Load trained theta and trainParams from approprite file.
% 3. Modify parameters below.
% 4. Run script.

%%% Parameters

dataset = 'huang'; % Word dataset used for training
zeroCategories = [4, 10]; % Cat, truck (by default, as used in the paper)
confusionCategories = [ 4, 10 ]; % Categories to use in nearest neighbor search.
                                 % May or may not be equal to zeroCategories.
numDistractors = 0:2:40; % Sequence of distractor word numbers (x-axis on graph)

%%% Begin code

addpath toolbox/pwmetric/;

disp('Loading random words for evaluation');
ee = load(['word_data/' dataset '/embeddings.mat']);
vv = load(['word_data/' dataset '/vocab.mat']);

load image_data/images/cifar10/meta.mat;
load(['word_data/' dataset '/cifar10/wordTable.mat']);

Xt = testX(:, testY == 10);
Yt = testY(testY == 10);
mX = mapDoMap(Xt, theta, trainParams);

accuracies = zeros(length(confusionCategories)+1, length(numDistractors));
for tt = 1:length(confusionCategories)
    confusionWordIds = knnsearch(ee.embeddings', wordTable(:, confusionCategories(tt))', 'K', 100);
    confusionWords_trainedRemoved = confusionWordIds(:, ~ismember(vv.vocab(confusionWordIds), label_names));
    fprintf('Neighbors for %s\n', label_names{confusionCategories(tt)});
    disp(vv.vocab(confusionWords_trainedRemoved(1:30)));
    for j = 1:length(numDistractors)
        words = [ wordTable(:, zeroCategories) ee.embeddings(:, confusionWords_trainedRemoved(1:numDistractors(j)))];
        tDist = slmetric_pw(words, mX, 'eucdist');
        [~, tGuessedCategories ] = min(tDist);
        candidateIds = ismember(tGuessedCategories, 1:length(zeroCategories));
        accuracies(tt, j) = sum(Yt(candidateIds) == zeroCategories(tGuessedCategories(candidateIds))) / length(Yt);
    end
end

% Random
fprintf('Random\n');
confusionWordIds = randi(length(vv.vocab), 1, 100);
confusionWords_trainedRemoved = confusionWordIds(:, ~ismember(vv.vocab(confusionWordIds), label_names));
disp(vv.vocab(confusionWords_trainedRemoved(1:30)));
for j = 1:length(numDistractors)
    words = [ wordTable(:, zeroCategories) ee.embeddings(:, confusionWords_trainedRemoved(1:numDistractors(j)))];
    tDist = slmetric_pw(words, mX, 'eucdist');
    [~, tGuessedCategories ] = min(tDist);
    candidateIds = ismember(tGuessedCategories, 1:length(zeroCategories));
    accuracies(length(confusionCategories)+1, j) = sum(Yt(candidateIds) == zeroCategories(tGuessedCategories(candidateIds))) / length(Yt);
end

markers = { '-+r', '-ob', '-dm' };
hold on;
for i = 1:length(confusionCategories)+1
    plot(numDistractors, accuracies(i, :), markers{i}, 'linewidth', 2);
end
h_legend = legend([ arrayfun(@(x) ['Neighbors of ', char(x)], label_names(confusionCategories), 'UniformOutput', false)', 'Random']);
h_xl = xlabel('Number of distractor words');
h_yl = ylabel('Accuracy');
set(h_legend, 'FontSize', 16);
set(h_xl, 'FontSize', 24);
set(h_yl, 'FontSize', 24);
hold off;

set(gcf,'paperunits','centimeters')
set(gcf,'papersize',[21,20]) % Desired outer dimensionsof figure
set(gcf,'paperposition',[0,0,21,20]) % Place plot on figure
