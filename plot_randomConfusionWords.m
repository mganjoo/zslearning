% Generates confusion words not already present in the dataset
% (because the system is trained well for classes it's already seen)

% Load test.mat
% Load theta and trainParams from appropriate file and modify 'dataset'
% based on how it was trained

addpath toolbox/pwmetric/;

dataset = 'huang';
zeroCategories = [4 10];

disp('Loading random words for evaluation');
ee = load(['word_data/' dataset '/embeddings.mat']);
vv = load(['word_data/' dataset '/vocab.mat']);

load image_data/images/cifar10/meta.mat;
load(['word_data/' dataset '/cifar10/wordTable.mat']);

Xt = testX(:, testY == 10);
Yt = testY(testY == 10);
% Xt = testX(:, ismember(testY, zeroCategories));
% Yt = testY(ismember(testY, zeroCategories));
mX = mapDoMap(Xt, theta, trainParams);

confusionCategories = [ 4, 10 ];
numRandom = 0:2:40;
accuracies = zeros(length(confusionCategories)+1, length(numRandom));
for tt = 1:length(confusionCategories)
    confusionWordIds = knnsearch(ee.embeddings', wordTable(:, confusionCategories(tt))', 'K', 100);
    confusionWords_trainedRemoved = confusionWordIds(:, ~ismember(vv.vocab(confusionWordIds), label_names));
    fprintf('Neighbors for %s\n', label_names{confusionCategories(tt)});
    disp(vv.vocab(confusionWords_trainedRemoved(1:30)));
    for j = 1:length(numRandom)
        words = [ wordTable(:, zeroCategories) ee.embeddings(:, confusionWords_trainedRemoved(1:numRandom(j)))];
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
for j = 1:length(numRandom)
    words = [ wordTable(:, zeroCategories) ee.embeddings(:, confusionWords_trainedRemoved(1:numRandom(j)))];
    tDist = slmetric_pw(words, mX, 'eucdist');
    [~, tGuessedCategories ] = min(tDist);
    candidateIds = ismember(tGuessedCategories, 1:length(zeroCategories));
    accuracies(length(confusionCategories)+1, j) = sum(Yt(candidateIds) == zeroCategories(tGuessedCategories(candidateIds))) / length(Yt);
end

markers = { '-+r', '-ob', '-dm' };
hold on;
for i = 1:length(confusionCategories)+1
    plot(numRandom, accuracies(i, :), markers{i}, 'linewidth', 2);
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
