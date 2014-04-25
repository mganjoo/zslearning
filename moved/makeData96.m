% Prepare the data

disp('Loading data');
train = load('image_data/features/cifar96/train.mat');
test = load('image_data/features/cifar96/test.mat');

disp('Creating training set');
makeFeatureBatches(train.trainX, train.trainY, 1, 'cifar96', { 'boy', 'lion', 'orange', 'camel' }, 'default_batch');

disp('Creating mini-training set');
makeFeatureBatches(train.trainX, train.trainY, 1, 'cifar96', { 'boy', 'lion', 'orange', 'camel' }, 'mini_batch', false, 2);

disp('Creating zero-shot training set');
makeFeatureBatches(train.trainX, train.trainY, 1, 'cifar96', { 'boy' }, 'zeroshot_batch', true);

disp('Creating zero-shot validation set');
makeTestBatch(test.testX, test.testY, 'cifar96', { 'boy', 'lion' }, 'zeroshot_test_batch', true);

clear train test;
