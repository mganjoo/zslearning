function [ categoryNames ] = loadCategoryNames( excludedCategories, dataset )

if nargin < 1
    excludedCategories = {};
end

b = load(['image_data/images/' dataset '/meta.mat']);
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

