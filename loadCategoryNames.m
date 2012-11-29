function [ categoryNames ] = loadCategoryNames( excludedCategories, dataset )

if nargin < 1
    excludedCategories = {};
end

if strcmp(dataset, 'cifar10') == true
    b = load('image_data/cifar-10-batches-mat/batches.meta.mat');
    allCategoryNames = b.label_names;
elseif strcmp(dataset, 'cifar100') == true
    b = load('image_data/cifar-100-matlab/meta96.mat');
    allCategoryNames = b.fine_label_names;
end

categoryNames = cell(length(allCategoryNames) - length(excludedCategories), 1);
k = 1;
for i = 1:length(allCategoryNames)
    if not(ismember(allCategoryNames{i}, excludedCategories))
        categoryNames{k} = allCategoryNames{i};
        k = k + 1;
    end
end

end

