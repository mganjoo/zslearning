% Prepare the data

disp('Loading data');
train = load('image_data/features/cifar96/train.mat');
test = load('image_data/features/cifar96/test.mat');

disp('Creating 9.2k training set');
makeFeatureBatches(train.trainX, train.trainY, 5, 'cifar96', { 'boy', 'lion', 'orange', 'camel' }, '9200_batch');

disp('Creating 4.6k training set');
makeFeatureBatches(train.trainX, train.trainY, 10, 'cifar96', { 'boy', 'lion', 'orange', 'camel' }, '4600_batch');

disp('Creating mini-training set');
makeFeatureBatches(train.trainX, train.trainY, 5, 'cifar96', { 'boy', 'lion', 'orange', 'camel' }, 'mini_batch', 2);

disp('Creating zero-shot validation set');
makeTestBatch(test.testX, test.testY, { 'boy' }, 'zeroshot_test_batch', true);

clear train test;
