addpath ../toolbox/;
addpath ../toolbox/minFunc/;

%% Shared parameters
fields = {{'patchDim',          8};      % patch dimension
          {'imageChannels',     3};      % number of channels (rgb, so 3)
          {'hiddenSize',        50};     % number of hidden units
};

for i = 1:length(fields)
    if exist('autoencParams','var') && isfield(autoencParams,fields{i}{1})
        disp(['Warning, we use the previously defined parameter ' fields{i}{1}])
    else
        autoencParams.(fields{i}{1}) = fields{i}{2};
    end
end

autoencParams.f = @sigmoid;
autoencParams.f_prime = @sigmoid_prime;

patchesPerCategory = 10000;        % patches in each category of image
visibleSize = autoencParams.patchDim ... % number of input units 
                * autoencParams.patchDim ...
                * autoencParams.imageChannels;
outputSize  = visibleSize;        % number of output units
hiddenSize  = autoencParams.hiddenSize; % number of hidden units 

epsilonZCA = 0.1;    % epsilon for ZCA whitening
sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term


%% STEP 0: Check gradient
debugHiddenSize = 5;
debugvisibleSize = 8;
patches = rand([8 10]);
theta = initializeParameters(debugHiddenSize, debugvisibleSize); 

[~, grad] = sparseAutoencoderCost(theta, debugvisibleSize, debugHiddenSize, ...
                                           lambda, sparsityParam, beta, ...
                                           autoencParams, patches);

% Check gradients
numGrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, debugvisibleSize, debugHiddenSize, ...
                                                  lambda, sparsityParam, beta, ...
                                                  autoencParams, patches), theta);

% Use this to visually compare the gradients side by side
disp([numGrad grad]); 

diff = norm(numGrad-grad)/norm(numGrad+grad);
disp(diff); 

assert(diff < 1e-9, 'Difference too large. Check your gradient computation again');

%% STEP a: Load patches
disp([ 'Loading ' num2str(patchesPerCategory) ' patches per category' ]);
patches = sampleImages(autoencParams, patchesPerCategory);
numPatches = size(patches, 2);

%% STEP b: Apply preprocessing

% Remove DC (mean of images). 
meanPatch = mean(patches, 2);
patches = bsxfun(@minus, patches, meanPatch);

% Apply ZCA whitening
sigma = patches * patches' / numPatches;
[u, s, ~] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilonZCA)) * u';
patches = ZCAWhite * patches;
displayColorNetwork(patches(:, 1:100));
pause;

%% STEP c: Learn features
disp('Learning');
theta = initializeParameters(hiddenSize, visibleSize);

options = struct;
options.Method = 'lbfgs'; 
options.maxIter = 400;
options.display = 'on';

[optTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, autoencParams, patches), ...
                              theta, options);

% Save the learned features and the preprocessing matrices for later use
fprintf('Saving learned features and preprocessing matrices...\n');                          
save('../savedParams/autoencoderParams.mat', 'optTheta', 'ZCAWhite', 'meanPatch', 'autoencParams');
fprintf('Saved\n');

%% STEP d: Visualize learned features

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
displayColorNetwork( (W*ZCAWhite)');
