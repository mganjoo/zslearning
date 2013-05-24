function [] = animals_split(word_dataset)

basedir = 'image_data/images/animals/';
s = load([basedir 'data.mat']);
load([basedir 'meta_all.mat']);

load(['word_data/' word_dataset '/vocab.mat']);
load(['word_data/' word_dataset '/embeddings.mat']);

tt = textscan(fopen([basedir 'trainclasses.txt']), '%s', 'delimiter', '\n');
trainclasses = tt{1};
tt = textscan(fopen([basedir 'testclasses.txt']), '%s', 'delimiter', '\n');
testclasses = tt{1};

trainIdxs = [];
testIdxs = [];
keptLabels = [];
disp('****Seen');
train_count = 0;
test_count = 0;
for i = 1:length(trainclasses)
    id = find(ismember(label_names, trainclasses{i}));
    if ~any(ismember(vocab, label_names{id}))
        continue;
    else
        keptLabels = [ keptLabels id ];
    end
    ttrain = floor(0.87 * sum(s.labels == id));
    ttest = sum(s.labels == id) - ttrain;
    fprintf('%s train: %d, ', char(label_names{id}), ttrain);
    fprintf('test: %d\n', ttest);
    train_count = train_count + ttrain;
    test_count = test_count + ttest;
    
    idxs = find(s.labels == id);
    idxs_order = randperm(length(idxs));
    
    trainIdxs = [ trainIdxs idxs(idxs_order(1:ttrain)) ];
    testIdxs = [ testIdxs idxs(idxs_order(ttrain+1:end)) ];
end

fprintf('seen test: %d\n', test_count);
disp('****Unseen');
unseenKeptLabels = [];
for i = 1:length(testclasses)
    id = find(ismember(label_names, testclasses{i}));
    if ~any(ismember(vocab, label_names{id}))
        continue;
    else
        keptLabels = [ keptLabels id ];
        unseenKeptLabels = [ unseenKeptLabels id ];
    end
    tt = sum(s.labels == id);
    fprintf('%s test: %d\n', char(label_names{id}), tt); 
    test_count = test_count + tt;

    idxs = find(s.labels == id);
    testIdxs = [ testIdxs idxs ];
end

fprintf('total train: %d\n', train_count);
fprintf('total test: %d\n', test_count);

mappedLabelIdxs = zeros(1, length(label_names));
c = 1;
keptLabels = unique(keptLabels);
for i = 1:length(keptLabels)
    mappedLabelIdxs(keptLabels(i)) = c;
    c = c + 1;
end

if not(exist(basedir, 'dir'))
    mkdir(basedir);
end

% Train
data = s.data(:, trainIdxs);
labels = mappedLabelIdxs(s.labels(trainIdxs));
save([basedir 'train.mat'], 'data', 'labels');

% Test
data = s.data(:, testIdxs);
labels = mappedLabelIdxs(s.labels(testIdxs));
save([basedir 'test.mat'], 'data', 'labels');

label_names = label_names(keptLabels);
zero_label_names = label_names(unseenKeptLabels);
save([basedir 'meta.mat'], 'label_names');
save([basedir 'zero.mat'], 'zero_label_names');

wordTable = zeros(size(embeddings, 1), length(label_names));
for i = 1:length(label_names)
    wordTable(:, i) = embeddings(:, ismember(vocab, label_names{i}));
end

word_outputdir = ['word_data/' word_dataset '/animals'];
if not(exist(word_outputdir, 'dir'))
    mkdir(word_outputdir);
end
save([word_outputdir '/wordTable.mat'], 'wordTable', 'label_names');

end