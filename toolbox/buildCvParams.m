function [ paramSets ] = buildCvParams(cvParams)
%BUILDCVPARAMS Summary of this function goes here
%   Detailed explanation goes here

paramLists = (cellfun(@(x) x{2}, cvParams, 'UniformOutput', false));
combinations = allcomb(paramLists{:});

paramSets = repmat(struct, 1, length(combinations));
for i = 1:length(combinations)
    for j = 1:length(cvParams)
        paramSets(i).(cvParams{j}{1}) = combinations(i, j);
    end
end

end

