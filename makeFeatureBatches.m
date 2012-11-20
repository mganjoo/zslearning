function [] = makeFeatureBatches(trainX, trainY, numBatches, excludedCategories, prefix, numPerCategory)

% Load categories
b = load('image_data/cifar-10-batches-mat/batches.meta.mat');
categoryNames = b.label_names;

% CIFAR constants (do not modify)
TOTAL_IMAGES = 50000;
totalCategories = length(categoryNames);
assert(mod(TOTAL_IMAGES, numBatches) == 0);
assert(mod(TOTAL_IMAGES / numBatches, totalCategories) == 0);
MAX_NUM_PER_CATEGORY = TOTAL_IMAGES / numBatches / totalCategories;

if nargin < 6
    numPerCategory = MAX_NUM_PER_CATEGORY;
    if nargin < 5
        prefix = 'default_batch';
        if nargin < 4
            excludedCategories = {};
        else
            disp('Excluded categories:');
            for i = 1:length(excludedCategories)
                disp(excludedCategories{i});
            end
        end
    end
end

assert(numPerCategory <= MAX_NUM_PER_CATEGORY);
fprintf('Putting %d samples per category in each file\n', numPerCategory);

numCategories = totalCategories - length(excludedCategories);
batches = cell(numBatches, 1);
for j = 1:numBatches
    batches{j} = zeros(1, numPerCategory * numCategories);
end
k = 1;
categorySet = zeros(1, numCategories);
for i = 1:totalCategories
    if ismember(categoryNames(i), excludedCategories)
        continue;
    end
    categorySet(k) = i;
    temp = find(trainY == i);
    for j = 1:numBatches
        batches{j}((k-1)*numPerCategory+1:k*numPerCategory) = temp((j-1)*numPerCategory+1:j*numPerCategory);
    end
    k = k + 1;
end

% Create directory if it doesn't exist
if not(exist('image_data/cifar-10-features', 'dir'))
    mkdir('image_data/cifar-10-features');
end

for i = 1:numBatches
    batches{i} = batches{i}(randperm(length(batches{i})));
    fprintf('Output batch %d\n', i);
    t = matfile(sprintf('image_data/cifar-10-features/%s_%d.mat', prefix, i));
    t.trainX = trainX(:, batches{i});
    trainYc = arrayfun(@(x) find(categorySet == x), trainY(:, batches{i}));
    t.trainY = trainYc;
    t.names = categoryNames(categorySet);
end

end
