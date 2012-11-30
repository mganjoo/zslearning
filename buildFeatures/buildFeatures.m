function [] = buildFeatures(dataset)

OUTPUT_DIR = ['../image_data/features/' dataset];
CIFAR_DIR  = ['../image_data/images/' dataset];

%%%%% Configuration
addpath ../toolbox/minFunc;
rfSize = 6;
numBases=1600;
CIFAR_DIM=[32 32 3];
alpha = 0.25;  %% CV-chosen value for soft-threshold function.
encoder='thresh';
encParam=alpha; % Use soft threshold encoder.

%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/train.mat']);

trainX = double(f1.data');
trainY = double(f1.labels);
clear f1;

% extract random patches
numPatches = 400000;
patches = zeros(numPatches, rfSize*rfSize*3);
for i=1:numPatches
  if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);
  patch = reshape(trainX(random('unid', size(trainX,1)),:), CIFAR_DIM);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
  patches(i,:) = patch(:)';
end

% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% ZCA whitening (with low-pass)
C = cov(patches);
M = mean(patches);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
patches = bsxfun(@minus, patches, M) * P;

% run training
dictionary = run_omp1(patches, numBases, 50);

% extract training features
trainXC = extract_features(trainX, dictionary, rfSize, CIFAR_DIM, M,P, encoder, encParam);

% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainX   = trainXCs';

%% Load CIFAR test data
fprintf('Loading test data...\n');
f1 = load([CIFAR_DIR '/test96.mat']);
testX = double(f1.data');
testY = double(f1.labels);
clear f1;

% compute testing features and standardize
testXC  = extract_features(testX, dictionary, rfSize, CIFAR_DIM, M,P, encoder, encParam);
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testX   = testXCs';

% save files
fprintf('Saving train data...\n');
save([OUTPUT_DIR 'train.mat'], 'trainX', 'trainY', '-v7.3');
fprintf('Saving test data...\n');
save([OUTPUT_DIR 'test.mat'], 'testX', 'testY', '-v7.3');

end
