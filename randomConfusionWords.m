% Generates confusion words not already present in the dataset
% (because the system is trained well for classes it's already seen)

% Load mX, Y, id_cat or id_truck and set zeroCategories
% Load theta and trainParams

addpath toolbox/pwmetric/;

disp('Loading random words for evaluation');
ee = load(['word_data/acl/embeddings.mat']);
vv = load(['word_data/acl/vocab.mat']);

load image_data/images/cifar10/meta.mat;
load word_data/acl/cifar10/wordTable.mat;
zeroCategories = [ 10 ];
id_old = id_truck;

ind = find(ismember(vv.vocab, label_names));
numRandom = 5:5:50;
accuracies = zeros(1, length(numRandom));
for j = 1:length(numRandom)
%     randIndices = randi(length(ind), 1, numRandom(j));
%     words = [ wordTable(:, zeroCategories) ee.embeddings(:, randIndices) ];
    id_new = id_old(~ismember(vv.vocab(id_old), label_names));    
    words = [ wordTable(:, zeroCategories) ee.embeddings(:, id_new(1:numRandom(j)))];
    tDist = slmetric_pw(words, mX, 'eucdist');
    [~, tGuessedCategories ] = min(tDist);
    candidateIds = ismember(tGuessedCategories, 1:length(zeroCategories));

    accuracies(j) = sum(Y(candidateIds) == zeroCategories(tGuessedCategories(candidateIds))) / length(Y);
    fprintf('Num_random: %d, Accuracy: %.3f\n', numRandom(j), accuracies(j));
end

plot(numRandom, accuracies);
