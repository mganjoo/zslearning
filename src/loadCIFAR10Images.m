function [ data, categories, categoryNames ] = loadCIFAR10Images( cifar_dir, trainParams, type )

numTrainingPerCat = 200;    % number of training examples per category
numValidationPerCat = 50;   % number of validation examples per category
numMiniPerCat = 2;          % number of mini (quick checking) examples per category

f1=load([cifar_dir '/data_batch_1.mat']);
f2=load([cifar_dir '/data_batch_2.mat']);
f3=load([cifar_dir '/data_batch_3.mat']);
f4=load([cifar_dir '/data_batch_4.mat']);
f5=load([cifar_dir '/data_batch_5.mat']);

images = double([f1.data; f2.data; f3.data; f4.data; f5.data])';
labels = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels])' + 1; % add 1 to labels!
clear f1 f2 f3 f4 f5;
featureSize = size(images, 1);

f = load([cifar_dir '/batches.meta.mat']);
categoryNames = f.label_names;
clear f;
numCategories = length(categoryNames) - length(trainParams.excludedCategories);

if strcmp(type, 'mini') == 1
    finalImgAndCats = zeros(featureSize+1, numMiniPerCat*numCategories);
elseif strcmp(type, 'validate') == 1
    finalImgAndCats = zeros(featureSize+1, numValidationPerCat*numCategories);
else
    finalImgAndCats = zeros(featureSize+1, numTrainingPerCat*numCategories);
end
k = 1;
categorySet = zeros(1, numCategories);
for i = 1:length(categoryNames)
    if ismember(categoryNames(i), trainParams.excludedCategories)
        continue;
    end
    categorySet(k) = i;
    temp = images(:, labels == i);
    if strcmp(type, 'mini') == 1
        temp = temp(:, 1:numMiniPerCat);
    elseif strcmp(type, 'validate') == 1
        temp = temp(:, numTrainingPerCat+1:numTrainingPerCat+numValidationPerCat);
    else % default to train
        temp = temp(:, 1:numTrainingPerCat);
    end
    % we now use one-based indexing
    finalImgAndCats(:, (k-1)*size(temp, 2)+1:k*size(temp, 2)) = [ temp; repmat(i, 1, size(temp, 2)) ];
    k = k + 1;
end

finalImgAndCats = finalImgAndCats(:,randperm(size(finalImgAndCats,2)));
data = finalImgAndCats(1:featureSize,:);
categories = arrayfun(@(x) find(categorySet == x), finalImgAndCats(end,:));
categoryNames = categoryNames(categorySet);

% Scale to [0, 1]
data = (data - min(data(:))) / (max(data(:)) - min(data(:)));

end
