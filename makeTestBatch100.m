function [] = makeTestBatch100(testX, testY, excludedCategories, prefix, noExclude, numPerCategory)

% Load categories
b = load('image_data/cifar-100-matlab/meta96.mat');
categoryNames = b.fine_label_names;

% CIFAR constants (do not modify)
TOTAL_IMAGES = 9600;
totalCategories = length(categoryNames);
MAX_NUM_PER_CATEGORY = TOTAL_IMAGES / totalCategories;

if nargin < 6
    numPerCategory = MAX_NUM_PER_CATEGORY;
    if nargin < 5
        noExclude = false; % noExclude can be set to true if the set of "excludedCategories" is actually the entire set of categories to use
        if nargin < 4
            prefix = 'default_test_batch_96';
            if nargin < 3 
                excludedCategories = {};
            else
                if noExclude == true
                    disp('Included categories:');
                else
                    disp('Excluded categories:');
                end
                for i = 1:length(excludedCategories)
                    disp(excludedCategories{i});
                end
            end
        end
    end
end
    
assert(numPerCategory <= MAX_NUM_PER_CATEGORY);
fprintf('Putting %d samples per category in each file\n', numPerCategory);

if noExclude == true
    numCategories = length(excludedCategories);
else
    numCategories = totalCategories - length(excludedCategories);
end
batch = zeros(1, numPerCategory * numCategories);
k = 1;
categorySet = zeros(1, numCategories);
for i = 1:totalCategories
    if noExclude == true && not(ismember(categoryNames(i), excludedCategories))
        continue;
    end
    if noExclude == false && ismember(categoryNames(i), excludedCategories)
        continue;
    end
    categorySet(k) = i;
    temp = find(testY == i);
    batch((k-1)*numPerCategory+1:k*numPerCategory) = temp(1:numPerCategory);
    k = k + 1;
end

% Create directory if it doesn't exist
if not(exist('image_data/cifar-features', 'dir'))
    mkdir('image_data/cifar-features');
end

batch = batch(randperm(length(batch)));
disp('Output test batch');
t = matfile(sprintf('image_data/cifar-features/%s.mat', prefix));
t.X = testX(:, batch);
testYc = arrayfun(@(x) find(categorySet == x), testY(:, batch));
t.Y = testYc;
t.names = categoryNames(categorySet);

end
