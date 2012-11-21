function [ categoryNames ] = loadCategoryNames( excludedCategories )

if nargin < 1
    excludedCategories = {};
end

b = load('image_data/cifar-10-batches-mat/batches.meta.mat');
allCategoryNames = b.label_names;

categoryNames = cell(length(allCategoryNames) - length(excludedCategories), 1);
k = 1;
for i = 1:length(allCategoryNames)
    if not(ismember(allCategoryNames{i}, excludedCategories))
        categoryNames{k} = allCategoryNames{i};
        k = k + 1;
    end
end

end

