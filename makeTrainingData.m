% Prepare the data

train = load('image_data/train.mat');
test = load('image_data/test.mat');

disp('Creating training set');
makeFeatureBatches(train.trainX, train.trainY, 5, { 'cat', 'truck' });

disp('Creating mini-training set');
makeFeatureBatches(train.trainX, train.trainY, 5, { 'cat', 'truck' }, 'mini_batch', 4);

disp('Creating zero-shot test set');
makeTestBatch(test.testX, test.testY, { 'airplane', 'automobile', 'bird', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' }, 'zeroshot_test_batch');

clear train test;
