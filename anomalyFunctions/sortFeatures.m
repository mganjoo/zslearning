function [sortedFeatures, sortedLabels] = sortFeatures(imageFeatures, labels)

[sortedLabels,permutation] = sort(labels);
sortedFeatures = imageFeatures(:,permutation);

end
