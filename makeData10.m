% Prepare the data

disp('Loading data');
train = load('image_data/features/cifar10/train.mat');
test = load('image_data/features/cifar10/test.mat');

disp('Creating training set');
makeFeatureBatches(train.trainX, train.trainY, 1, 'cifar10', { 'cat', 'truck' }, 'default_batch');

disp('Creating mini-training set');
makeFeatureBatches(train.trainX, train.trainY, 1, 'cifar10', { 'cat', 'truck' }, 'mini_batch', false, 2);

disp('Creating zero-shot training set');
makeFeatureBatches(train.trainX, train.trainY, 1, 'cifar10', { 'cat' }, 'zeroshot_batch', true);

disp('Creating zero-shot validation set');
makeTestBatch(test.testX, test.testY, 'cifar10', { 'cat', 'truck' }, 'zeroshot_test_batch', true);

clear train test;
