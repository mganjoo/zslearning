% Prepare the data

disp('Loading data');
train = load('image_data/train.mat');
test = load('image_data/test.mat');

disp('Creating 8k training set');
makeFeatureBatches(train.trainX, train.trainY, 5, { 'cat', 'truck' }, '8k_batch');

disp('Creating 4k training set');
makeFeatureBatches(train.trainX, train.trainY, 10, { 'cat', 'truck' }, '4k_batch');

disp('Creating mini-training set');
makeFeatureBatches(train.trainX, train.trainY, 5, { 'cat', 'truck' }, 'mini_batch', 4);

disp('Creating zero-shot validation set');
makeTestBatch(test.testX, test.testY, { 'cat' }, 'zeroshot_test_batch', true);

clear train test;
