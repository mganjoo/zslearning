function patches = sampleImages(autoencParams, patchesPerCategory)

addpath ../toolbox/;

% Load CIFAR-10 images
[imgs, categories, allCategories, categoryLabels] = loadCIFAR10Images('../image_data/cifar-10-batches-mat', 'train', { 'cat', 'truck' });
numCategories = length(allCategories);

patches = zeros(autoencParams.patchDim*autoencParams.patchDim*autoencParams.imageChannels, patchesPerCategory*numCategories);

categorizedImgs = cell(length(categoryLabels), 1);
for i = 1:numCategories
    categorizedImgs{allCategories(i)} = imgs(:, categories == allCategories(i));
end

ipatch = 1;
for i = 1:numCategories
    data = categorizedImgs{allCategories(i)};
    ilength = sqrt(size(data, 1) / 3);
    for j = 1:patchesPerCategory
        % randomly select an image from the subset
        icolumn = data(:, randi(size(data, 2)));
        hstart = randi(ilength - autoencParams.patchDim + 1);
        hend = hstart + (autoencParams.patchDim - 1);
        vstart = randi(ilength - autoencParams.patchDim + 1);
        vend = vstart + (autoencParams.patchDim - 1);
        indices = reshape(bsxfun(@plus, ilength * (0:(vend - vstart))', repmat(hstart:hend, vend - vstart + 1, 1)), 1, []);
        % convert to 3-color indices
        cindices = [ indices, indices + ilength*ilength, indices + ilength*ilength*2 ];
        patches(:,ipatch) = icolumn(cindices);
        ipatch = ipatch + 1;
    end
end

patches = patches(:, randperm(size(patches,2)));

end
