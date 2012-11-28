function [] = makeTestBatch(testX, testY, excludedCategories, prefix, numPerCategory)

% Load categories
b = load('image_data/cifar-10-batches-mat/batches.meta.mat');
categoryNames = b.label_names;

% CIFAR constants (do not modify)
TOTAL_IMAGES = 10000;
totalCategories = length(categoryNames);
MAX_NUM_PER_CATEGORY = TOTAL_IMAGES / totalCategories;

if nargin < 5
    numPerCategory = MAX_NUM_PER_CATEGORY;
    if nargin < 4
        prefix = 'default_test_batch';
        if nargin < 3 
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
batch = zeros(1, numPerCategory * numCategories);
k = 1;
categorySet = zeros(1, numCategories);
for i = 1:totalCategories
    if ismember(categoryNames(i), excludedCategories)
        continue;
    end
    categorySet(k) = i;
    temp = find(testY == i);
    batch((k-1)*numPerCategory+1:k*numPerCategory) = temp(1:numPerCategory);
    k = k + 1;
end

% Create directory if it doesn't exist
if not(exist('image_data/cifar-10-features', 'dir'))
    mkdir('image_data/cifar-10-features');
end

batch = batch(randperm(length(batch)));
disp('Output test batch');
t = matfile(sprintf('image_data/cifar-10-features/%s.mat', prefix));
t.X = testX(:, batch);
testYc = arrayfun(@(x) find(categorySet == x), testY(:, batch));
t.Y = testYc;
t.names = categoryNames(categorySet);

end
