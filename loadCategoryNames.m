function [ categoryNames ] = loadCategoryNames( excludedCategories )

if nargin < 1
    excludedCategories = {};
end

b = load('image_data/cifar-100-matlab/meta96.mat');
allCategoryNames = b.fine_label_names;

categoryNames = cell(length(allCategoryNames) - length(excludedCategories), 1);
k = 1;
for i = 1:length(allCategoryNames)
    if not(ismember(allCategoryNames{i}, excludedCategories))
        categoryNames{k} = allCategoryNames{i};
        k = k + 1;
    end
end

end

