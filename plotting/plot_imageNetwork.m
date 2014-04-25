addpath toolbox/pwmetric/;
addpath toolbox;

IMAGES_PER_CATEGORY = 3;

load('image_data/images/cifar10/test.mat')
load('image_data/features/cifar10/test.mat');
mappedTestImages = mapDoMap(testX, thetaMapping, mapTrainParams);
dists = slmetric_pw(wordTable, mappedTestImages, 'eucdist');

chosen_categories = [4, 5, 7, 10];
selected = [1, 500, 3000, 8000, 10000];
image_idxs = zeros(1, length(chosen_categories) * length(selected));
k = 1;
for i = 1:length(chosen_categories)
    [~, idxs] = sort(dists(chosen_categories(i), :));
    for j = 1:length(selected)
        image_idxs(k) = idxs(selected(j));
        k = k + 1;
    end
end

image_idxs = unique(image_idxs);
displayColorNetwork(double(data(:, image_idxs)));

closestWords = cell(10, length(image_idxs));
for i = 1:length(image_idxs)
    [~, idxs] = sort(dists(:, i));
    closestWords(:, i) = label_names(idxs);
end
