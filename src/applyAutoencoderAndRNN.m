function [ out, outVectorLength ] = applyAutoencoderAndRNN(imgs, WbAutoenc, meanPatch, ZCAWhite, WbRNN, autoencParams, trainParams)
    if isfield(trainParams, 'disableAutoencoder') && trainParams.disableAutoencoder == true
        out = imgs;
        outVectorLength = size(imgs, 1);
        return;
    end

    numImages = size(imgs, 2);
    assert(mod(trainParams.imageDim, autoencParams.patchDim) == 0);
    numPatches = (trainParams.imageDim*trainParams.imageDim) / (autoencParams.patchDim*autoencParams.patchDim);
    patches = cell(numPatches, 1);
    ipatch = 1;
    for i = 1:sqrt(numPatches)
        vstart = (i-1)*autoencParams.patchDim+1;
        vend = i*autoencParams.patchDim;
        for j = 1:sqrt(numPatches)
            hstart = (j-1)*autoencParams.patchDim+1;
            hend = j*autoencParams.patchDim;
            indices = reshape(bsxfun(@plus, trainParams.imageDim * (0:(vend - vstart))', repmat(hstart:hend, vend - vstart + 1, 1)), 1, []);
            % convert to 3-color indices
            cindices = [ indices, indices + trainParams.imageDim*trainParams.imageDim, indices + trainParams.imageDim*trainParams.imageDim*2 ];
            patches{ipatch} = imgs(cindices, 1:numImages);
            ipatch = ipatch + 1;
        end
    end
    
    % Apply autoencoder
    inRNN = cell(numPatches, 1);
    for i = 1:numPatches    
        patchGroup = ZCAWhite * (bsxfun(@minus, patches{i}, meanPatch));
        inRNN{i} = autoencParams.f(WbAutoenc * [ patchGroup; ones(1, numImages)]);
    end
    
    if isfield(trainParams, 'disableRNN') && trainParams.disableRNN == true
        out = cell2mat(inRNN);
        outVectorLength = size(out, 1);
        return;
    end
    
    % Apply RNN
    level2GroupSize = sqrt(numPatches);
    level2RNN = cell(level2GroupSize, 1);
    for i = 1:level2GroupSize
        level2RNN{i} = trainParams.f(WbRNN * [cell2mat(inRNN(i:i+level2GroupSize-1)); ones(1, numImages)]);
    end
    out = trainParams.f(WbRNN * [cell2mat(level2RNN); ones(1, numImages)]);
    outVectorLength = size(out, 1);
end