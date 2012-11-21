CIFAR_DIR='../image_data/cifar-10-batches-mat/';
OUTPUT_DIR='../image_data/';
SPAMS_DIR='/path/to/SPAMS/release/platform'; % E.g.: 'SPAMS/release/mkl64'

%%%%% Configuration
addpath ../toolbox/minFunc;
rfSize = 6;
numBases=1600;
CIFAR_DIM=[32 32 3];
alpha = 0.25;  %% CV-chosen value for soft-threshold function.
lambda = 1.0;  %% CV-chosen sparse coding penalty.

%%%%% Dictionary Training %%%%%%
%alg='patches'; %% Use randomly sampled patches.  Test accuracy 79.14%
alg='omp1';   %% Use 1-hot VQ (OMP-1).  Test accuracy 79.96%
%alg='sc';     %% Sparse coding

%%%%% Encoding %%%%%%
encoder='thresh'; encParam=alpha; %% Use soft threshold encoder.
%encoder='sc'; encParam=lambda; %% Use sparse coding for encoder.

%%%%% SVM Parameter %%%%%
switch (encoder)
 case 'thresh'
  L = 0.01; % L=0.01 for 1600 features.  Use L=0.03 for 4000-6000 features.
 case 'sc'
  L = 1.0; % May need adjustment for various combinations of training / encoding parameters.
end

%% Check SPAMS install
if (strcmp(alg,'sc') || strcmp(encoder, 'sc'))
  assert(~strcmp(SPAMS_DIR, '/path/to/SPAMS/release/platform'), ...
         ['You need to modify run_feature_extraction.m so that SPAMS_DIR points to ' ...
          'the SPAMS toolkit release directory.  You can download this ' ...
          'toolkit from:  http://www.di.ens.fr/willow/SPAMS/downloads.html']);
  addpath(SPAMS_DIR);
end

%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);

trainX = double([f1.data; f2.data; f3.data; f4.data; f5.data]);
trainY = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!
clear f1 f2 f3 f4 f5;

% extract random patches
switch (alg)
 case 'omp1'
  numPatches = 400000;
 case 'sc'
  numPatches = 100000;
 case 'patches'
  numPatches = 50000; % still needed for whitening
end
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
switch alg
 case 'omp1'
  dictionary = run_omp1(patches, numBases, 50);
 case 'sc'
  dictionary = run_sc(patches, numBases, 10, lambda);
 case 'patches'
    dictionary = patches(randsample(size(patches,1), numBases), :);
    dictionary = bsxfun(@rdivide, dictionary, sqrt(sum(dictionary.^2,2)) + 1e-20);
end

% extract training features
trainXC = extract_features(trainX, dictionary, rfSize, CIFAR_DIM, M,P, encoder, encParam);

% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainX   = trainXCs';
trainY   = trainY';

%% Load CIFAR test data
fprintf('Loading test data...\n');
f1 = load([CIFAR_DIR '/test_batch.mat']);
testX = double(f1.data);
testY = double(f1.labels) + 1;
clear f1;

% compute testing features and standardize
testXC  = extract_features(testX, dictionary, rfSize, CIFAR_DIM, M,P, encoder, encParam);
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testX   = testXCs';
testY   = testY';

% save files
save([OUTPUT_DIR 'train.mat'], 'trainX', 'trainY', '-v7.3');
save([OUTPUT_DIR 'test.mat'], 'testX', 'testY', '-v7.3');
