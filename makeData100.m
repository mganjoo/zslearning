% Prepare the data

disp('Loading data');
train = load('image_data/train100.mat');
test = load('image_data/test100.mat');

disp('Creating 9.2k training set');
makeFeatureBatches100(train.trainX, train.trainY_fine, 5, { 'boy', 'lion', 'orange', 'camel' }, '9200_batch_96');

disp('Creating 4.6k training set');
makeFeatureBatches100(train.trainX, train.trainY_fine, 10, { 'boy', 'lion', 'orange', 'camel' }, '4600_batch_96');

disp('Creating mini-training set');
makeFeatureBatches100(train.trainX, train.trainY_fine, 5, { 'boy', 'lion', 'orange', 'camel' }, 'mini_batch_96', 2);

disp('Creating zero-shot validation set');
makeTestBatch100(test.testX, test.testY_fine, { 'boy' }, 'zeroshot_test_batch_96', true);

clear train test;
