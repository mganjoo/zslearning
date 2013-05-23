function [seenIdxs, unseenIdxs, mappedY, numCategories] = animals_split(word_dataset, featuresName)

basedir = 'image_data/features/animals/';
load([basedir featuresName '.mat']);

load(['word_data/' word_dataset '/vocab.mat']);
load(['word_data/' word_dataset '/embeddings.mat']);

tt = textscan(fopen([basedir 'trainclasses.txt']), '%s', 'delimiter', '\n');
trainclasses = tt{1};
tt = textscan(fopen([basedir 'testclasses.txt']), '%s', 'delimiter', '\n');
testclasses = tt{1};

seenIdxs = [];
unseenIdxs = [];
keptLabels = [];
for i = 1:length(trainclasses)
    id = find(ismember(labels, trainclasses{i}));
    if ~any(ismember(vocab, labels{id}))
        continue;
    else
        keptLabels = [ keptLabels id ];
    end
    tt_train = find(Y == id);
    seenIdxs = [ seenIdxs tt_train ];
end

for i = 1:length(testclasses)
    id = find(ismember(labels, testclasses{i}));
    if ~any(ismember(vocab, labels{id}))
        continue;
    else
        keptLabels = [ keptLabels id ];
    end
    tt = find(Y == id);
    unseenIdxs = [ unseenIdxs tt ];
end

mappedY = Y;
for i = 1:length(Y)
    foundIdx = keptLabels == Y(i);
    if any(foundIdx)
        mappedY(i) = find(foundIdx); 
    else
        mappedY(i) = 0;
    end
end

label_names = labels(keptLabels);
wordTable = zeros(size(embeddings, 1), length(label_names));
for i = 1:length(label_names)
    wordTable(:, i) = embeddings(:, ismember(vocab, label_names{i}));
end

numCategories = length(label_names);
save([basedir featuresName '_idxs.mat'], 'seenIdxs', 'unseenIdxs', 'mappedY', 'numCategories');
outputDir = ['word_data/' word_dataset '/' featuresName];
if not(exist(outputDir, 'dir'))
    mkdir(outputDir);
end
save([outputDir '/wordTable.mat'], 'wordTable', 'label_names');

end